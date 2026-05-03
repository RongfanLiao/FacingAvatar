"""Batch motion diffusion inference with gen_vis-ready exports.

This dedicated entrypoint keeps the motion diffusion checkpoint and manifest in
memory for the whole run instead of invoking inference.py once per sequence.
Each completed sample is written to:

    <output_dir>/<seq_id>_right/flame_param.npz

alongside foreground_image.png and transforms.json when they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from decord import VideoReader, cpu as decord_cpu
from scipy.ndimage import uniform_filter1d

from benchmark.lookingface import _fit_frame_to_canvas
from benchmark.motion_diffusion import MotionDiffusionModel
from benchmark.motion_transvae import checkpoint_state_dict
from config import DEVICE, WAV2VEC_EMB_DIR
from dataset import FLAME_KEYS, _interpolate_features
from inference import (
    _detect_checkpoint_family,
    _infer_motion_diffusion_config,
    _load_checkpoint,
    _motion_prediction_to_flame,
    _stabilize_pose_tracks,
)
from manifest import load_manifest


DEFAULT_SPLIT_PATH = "data/LookingFace/dataset_splits/test.json"
DEFAULT_CHECKPOINT = "checkpoints/motion_diffusion_port/best.pt"
DEFAULT_OUTPUT_DIR = "output/test_gen_vis"
DEFAULT_INFERENCE_TIMESTEPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run motion diffusion inference over a LookingFace split and export gen_vis-ready directories"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Motion diffusion checkpoint")
    parser.add_argument("--split_path", type=str, default=DEFAULT_SPLIT_PATH, help="Path to predefined split JSON")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write gen_vis-ready outputs")
    parser.add_argument("--smooth", type=int, default=5, help="Temporal smoothing window size applied to predicted FLAME keys")
    parser.add_argument("--pose_noise_filter", type=int, default=8, help="Median-filter window used to remove rotation/translation spikes before export (<=1 disables it)")
    parser.add_argument("--pose_smooth", type=int, default=9, help="Additional smoothing window applied to rotation/translation before export (<=1 disables it)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run sequences that already have flame_param.npz outputs")
    parser.add_argument("--no_overwrite", action="store_false", dest="overwrite", help="Keep cached flame_param.npz outputs instead of regenerating them")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of split sequences to process")
    parser.add_argument("--start_index", type=int, default=0, help="Optional starting index into the resolved split sequence list")
    parser.add_argument("--video_canvas_size", type=int, default=300, help="Canvas size for left video frames before encoding")
    parser.add_argument("--zero_rotation_translation", action="store_true", help="Zero out rotation and translation in the exported output")
    parser.add_argument("--motion_diffusion_n_heads", type=int, default=8, help="Attention head count used when reconstructing the model")
    parser.add_argument("--motion_diffusion_dropout", type=float, default=0.1, help="Dropout used when reconstructing the model")
    parser.add_argument("--motion_diffusion_train_timesteps", type=int, default=1000, help="Training diffusion schedule length")
    parser.add_argument("--motion_diffusion_inference_timesteps", type=int, default=DEFAULT_INFERENCE_TIMESTEPS, help="Reverse diffusion steps used during inference")
    parser.add_argument("--motion_diffusion_beta_start", type=float, default=1e-4, help="Inference beta schedule start")
    parser.add_argument("--motion_diffusion_beta_end", type=float, default=2e-2, help="Inference beta schedule end")
    parser.add_argument("--motion_diffusion_clip_sample", type=float, default=5.0, help="Sample clamp range during inference")
    parser.add_argument("--motion_diffusion_guidance_scale", type=float, default=1.5, help="Classifier-free guidance scale during inference")
    parser.add_argument(
        "--motion_diffusion_timestep_spacing",
        choices=["leading", "linspace", "trailing", "full"],
        default="leading",
        help="Timestep spacing strategy used during inference",
    )
    parser.add_argument("--motion_diffusion_ddim_eta", type=float, default=0.0, help="DDIM eta used during inference")
    parser.set_defaults(overwrite=True)
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_seq_ids(split_path: Path) -> list[str]:
    with open(split_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    seq_ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.endswith("_left.mp4"):
            continue
        seq_id = Path(entry).stem.removesuffix("_left")
        if seq_id not in seen:
            seq_ids.append(seq_id)
            seen.add(seq_id)
    return seq_ids


def expected_output_path(output_dir: Path, seq_id: str) -> Path:
    return output_dir / f"{seq_id}_right" / "flame_param.npz"


def build_motion_inputs(entry: dict[str, str], seq_id: str, n_frames: int, video_canvas_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wav2vec_path = Path(WAV2VEC_EMB_DIR) / f"{seq_id}_left.npy"
    if not wav2vec_path.exists():
        raise FileNotFoundError(f"Missing wav2vec feature file: {wav2vec_path}")

    wav2vec_feat = np.load(wav2vec_path)
    wav2vec_feat = _interpolate_features(wav2vec_feat, n_frames)
    audio_tensor = torch.from_numpy(np.asarray(wav2vec_feat, dtype=np.float32)).float().unsqueeze(0)

    vr = VideoReader(entry["left_mp4"], ctx=decord_cpu(0))
    total_video_frames = len(vr)
    if n_frames > 0 and total_video_frames != n_frames:
        indices = np.linspace(0, total_video_frames - 1, n_frames).astype(int).tolist()
    else:
        indices = list(range(total_video_frames))

    frames = []
    for frame_index in indices:
        frame_np = vr[frame_index].asnumpy()
        frames.append(_fit_frame_to_canvas(frame_np, video_canvas_size))
    video_tensor = torch.stack(frames).unsqueeze(0)
    padding_mask = torch.zeros((1, n_frames), dtype=torch.bool)
    return audio_tensor, video_tensor, padding_mask


def build_model(args: argparse.Namespace, checkpoint_path: Path, device: torch.device) -> tuple[MotionDiffusionModel, dict[str, int]]:
    ckpt = _load_checkpoint(str(checkpoint_path), device)
    state_dict = checkpoint_state_dict(ckpt)
    checkpoint_family = _detect_checkpoint_family(state_dict)
    if checkpoint_family != "motion_diffusion":
        raise RuntimeError(f"Expected a motion diffusion checkpoint, got {checkpoint_family!r}")

    diffusion_config = _infer_motion_diffusion_config(state_dict)
    model = MotionDiffusionModel(
        audio_dim=int(diffusion_config["audio_dim"]),
        target_dim=int(diffusion_config["output_dim"]),
        feature_dim=int(diffusion_config["feature_dim"]),
        n_heads=args.motion_diffusion_n_heads,
        num_layers=int(diffusion_config["num_layers"]),
        dropout=args.motion_diffusion_dropout,
        train_timesteps=args.motion_diffusion_train_timesteps,
        inference_timesteps=args.motion_diffusion_inference_timesteps,
        beta_start=args.motion_diffusion_beta_start,
        beta_end=args.motion_diffusion_beta_end,
        clip_sample=args.motion_diffusion_clip_sample,
        guidance_scale=args.motion_diffusion_guidance_scale,
        audio_drop_prob=0.0,
        video_drop_prob=0.0,
        latent_drop_prob=0.0,
        timestep_spacing=args.motion_diffusion_timestep_spacing,
        ddim_eta=args.motion_diffusion_ddim_eta,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, diffusion_config


def export_sequence(
    seq_id: str,
    entry: dict[str, str],
    model: MotionDiffusionModel,
    output_dim: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    with np.load(entry["flame_npz"]) as label_npz:
        n_frames = int(label_npz["expr"].shape[0])
        audio_t, video_t, padding_mask = build_motion_inputs(entry, seq_id, n_frames, args.video_canvas_size)
        audio_t = audio_t.to(device)
        video_t = video_t.to(device)
        padding_mask = padding_mask.to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                prediction = model.sample(
                    left_audio_feat=audio_t,
                    left_video_frames=video_t,
                    padding_mask=padding_mask,
                )

        pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
        results = _motion_prediction_to_flame(pred_np, label_npz, output_dim)

        if args.smooth > 0:
            for key in FLAME_KEYS:
                results[key] = uniform_filter1d(results[key], size=args.smooth, axis=0, mode="nearest")

        _stabilize_pose_tracks(
            results,
            noise_window=args.pose_noise_filter,
            smooth_window=args.pose_smooth,
        )

        for key in label_npz.files:
            if key not in results:
                results[key] = np.asarray(label_npz[key])

        if args.zero_rotation_translation:
            for key in ("rotation", "translation"):
                if key in results:
                    results[key] = np.zeros_like(results[key])
                elif key in label_npz:
                    results[key] = np.zeros_like(label_npz[key])

    seq_out_dir = output_dir / f"{seq_id}_right"
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = seq_out_dir / "flame_param.npz"
    np.savez(out_path, **results)

    label_dir = Path(entry["flame_npz"]).parent
    for name in ("foreground_image.png", "transforms.json"):
        src = label_dir / name
        if src.exists():
            shutil.copy2(src, seq_out_dir / name)

    return out_path


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_repo_path(args.checkpoint)
    split_path = resolve_repo_path(args.split_path)
    output_dir = resolve_repo_path(args.output_dir)

    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    seq_ids = load_seq_ids(split_path)
    if args.start_index:
        seq_ids = seq_ids[args.start_index:]
    if args.limit is not None:
        seq_ids = seq_ids[:args.limit]
    if not seq_ids:
        print("No sequences resolved from the split file.")
        return

    manifest = load_manifest()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model, diffusion_config = build_model(args, checkpoint_path, device)
    output_dim = int(diffusion_config["output_dim"])

    completed = 0
    skipped = 0
    failed = 0

    print(f"Resolved {len(seq_ids)} sequences from {split_path}")
    print(f"Loaded manifest once: {len(manifest)} sequences")
    print(f"Loaded motion diffusion checkpoint once: {checkpoint_path}")
    for index, seq_id in enumerate(seq_ids, start=1):
        out_path = expected_output_path(output_dir, seq_id)
        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{len(seq_ids)}] Skipping {seq_id}: cached")
            skipped += 1
            continue

        entry = manifest.get(seq_id)
        if entry is None:
            failed += 1
            print(f"[{index}/{len(seq_ids)}] ERROR: {seq_id} not found in manifest")
            continue

        print(f"[{index}/{len(seq_ids)}] Running motion diffusion inference for {seq_id}")
        try:
            written_path = export_sequence(
                seq_id=seq_id,
                entry=entry,
                model=model,
                output_dim=output_dim,
                output_dir=output_dir,
                args=args,
                device=device,
            )
        except Exception as exc:
            failed += 1
            print(f"  ERROR: {exc}")
            continue

        completed += 1
        print(f"  Saved to {written_path}")

    print(
        f"Done: {completed} completed, {skipped} skipped, {failed} failed. "
        f"Outputs under {output_dir}"
    )


if __name__ == "__main__":
    main()