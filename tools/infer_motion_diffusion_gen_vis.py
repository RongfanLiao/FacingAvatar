"""Batch gen_vis inference with rendering-ready FLAME exports.

This dedicated entrypoint keeps a supported checkpoint and manifest in memory
for the whole run instead of invoking inference.py once per sequence.
Each completed sample is written to:

    <output_dir>/<seq_id>_right/flame_param.npz

alongside foreground_image.png and transforms.json when they exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from benchmark.dyadic_dim import DyadicContinuousTransformer
from benchmark.dualtalk import LookingFaceDualTalk
from benchmark.listenformer import LookingFaceListenFormer
from benchmark.motion_diffusion import MotionDiffusionModel
from benchmark.motion_flow_matching import MotionFlowMatchingModel
from benchmark.motion_transvae import MotionTransformerVAE, checkpoint_state_dict
from benchmark.regnn import LookingFaceREGNN
from config import DEVICE
from dataset import FLAME_KEYS
from inference import (
    _build_motion_transvae_inputs,
    _detect_checkpoint_family,
    _infer_listenformer_config,
    _infer_motion_diffusion_config,
    _infer_regnn_config,
    _load_checkpoint,
    _motion_prediction_to_flame,
    _resolve_motion_flow_matching_config,
    _resolve_dualtalk_config,
    _stabilize_pose_tracks,
)
from manifest import load_manifest


DEFAULT_SPLIT_PATH = "data/LookingFace/dataset_splits/test.json"
DEFAULT_CHECKPOINT = "checkpoints/motion_diffusion_port/best.pt"
DEFAULT_OUTPUT_DIR = "output/test_gen_vis"
DEFAULT_INFERENCE_TIMESTEPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference over a LookingFace split and export gen_vis-ready directories"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Supported checkpoint (motion diffusion, motion flow matching, motion TransVAE, Dyadic DIM, ListenFormer, REGNN, or DualTalk)")
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
    parser.add_argument("--motion_flow_matching_n_heads", type=int, default=8, help="Attention head count used when reconstructing motion flow matching")
    parser.add_argument("--motion_flow_matching_dropout", type=float, default=0.1, help="Dropout used when reconstructing motion flow matching")
    parser.add_argument("--motion_flow_matching_video_chunk_size", type=int, default=32, help="Video encoder chunk size used when reconstructing motion flow matching")
    parser.add_argument("--motion_flow_matching_solver", choices=["euler", "heun"], default="heun", help="ODE solver used during motion flow matching inference")
    parser.add_argument("--motion_flow_matching_solver_steps", type=int, default=40, help="ODE integration steps used during motion flow matching inference")
    parser.add_argument("--motion_flow_matching_clip_sample", type=float, default=5.0, help="Sample clamp range during motion flow matching inference")
    parser.add_argument("--motion_flow_matching_guidance_scale", type=float, default=1.5, help="Classifier-free guidance scale used during motion flow matching inference")
    parser.add_argument("--motion_flow_matching_time_embed_scale", type=float, default=1000.0, help="Continuous-time embedding scale used during motion flow matching inference")
    parser.add_argument("--listenformer_n_heads", type=int, default=8, help="Attention head count used when reconstructing ListenFormer")
    parser.add_argument("--listenformer_dropout", type=float, default=0.1, help="Dropout used when reconstructing ListenFormer")
    parser.add_argument("--listenformer_video_chunk_size", type=int, default=8, help="Video encoder chunk size used when reconstructing ListenFormer")
    parser.add_argument("--dyadic_n_heads", type=int, default=8, help="Attention head count used when reconstructing Dyadic DIM")
    parser.add_argument("--dyadic_dropout", type=float, default=0.1, help="Dropout used when reconstructing Dyadic DIM")
    parser.add_argument("--dyadic_video_chunk_size", type=int, default=8, help="Video encoder chunk size used when reconstructing Dyadic DIM")
    parser.add_argument("--dualtalk_n_heads", type=int, default=8, help="Attention head count used when reconstructing DualTalk")
    parser.add_argument("--dualtalk_dropout", type=float, default=0.1, help="Dropout used when reconstructing DualTalk")
    parser.add_argument("--dualtalk_video_chunk_size", type=int, default=8, help="Video encoder chunk size used when reconstructing DualTalk")
    parser.add_argument("--dualtalk_modulation_factor", type=float, default=0.1, help="Modulation factor used when reconstructing DualTalk")
    parser.add_argument("--regnn_neighbors", type=int, default=6, help="Neighbor count used when reconstructing REGNN")
    parser.add_argument("--regnn_act_type", choices=["ELU", "ReLU", "GeLU", "None"], default="ELU", help="Graph activation used when reconstructing REGNN")
    parser.add_argument("--regnn_noise_threshold", type=float, default=0.0, help="Optional REGNN noise threshold; <=0 disables it")
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


def build_model(
    args: argparse.Namespace,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int], str]:
    ckpt = _load_checkpoint(str(checkpoint_path), device)
    state_dict = checkpoint_state_dict(ckpt)
    checkpoint_family = _detect_checkpoint_family(state_dict)
    if checkpoint_family == "motion_diffusion":
        model_config = _infer_motion_diffusion_config(state_dict)
        model = MotionDiffusionModel(
            audio_dim=int(model_config["audio_dim"]),
            target_dim=int(model_config["output_dim"]),
            feature_dim=int(model_config["feature_dim"]),
            n_heads=args.motion_diffusion_n_heads,
            num_layers=int(model_config["num_layers"]),
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
    elif checkpoint_family == "motion_flow_matching":
        flow_config = _resolve_motion_flow_matching_config(ckpt, state_dict)
        model_config = dict(flow_config)
        saved_config = ckpt.get("model_config")
        video_chunk_size = args.motion_flow_matching_video_chunk_size
        if isinstance(saved_config, dict) and saved_config.get("family") == "motion_flow_matching":
            video_chunk_size = int(saved_config.get("video_chunk_size", video_chunk_size))
        model = MotionFlowMatchingModel(
            audio_dim=int(flow_config["audio_dim"]),
            target_dim=int(flow_config["output_dim"]),
            feature_dim=int(flow_config["feature_dim"]),
            n_heads=int(flow_config.get("n_heads", args.motion_flow_matching_n_heads)),
            num_layers=int(flow_config["num_layers"]),
            dropout=float(flow_config.get("dropout", args.motion_flow_matching_dropout)),
            video_chunk_size=video_chunk_size,
            solver=str(flow_config.get("solver", args.motion_flow_matching_solver)),
            solver_steps=int(flow_config.get("solver_steps", args.motion_flow_matching_solver_steps)),
            clip_sample=float(flow_config.get("clip_sample", args.motion_flow_matching_clip_sample)),
            guidance_scale=float(flow_config.get("guidance_scale", args.motion_flow_matching_guidance_scale)),
            audio_drop_prob=0.0,
            video_drop_prob=0.0,
            latent_drop_prob=0.0,
            time_embed_scale=float(flow_config.get("time_embed_scale", args.motion_flow_matching_time_embed_scale)),
        ).to(device)
    elif checkpoint_family == "motion_transvae":
        output_dim = int(state_dict["decoder.output_head.weight"].shape[0])
        model_config = {"output_dim": output_dim}
        model = MotionTransformerVAE(output_dim=output_dim).to(device)
    elif checkpoint_family == "dyadic_dim":
        encoder_layer_indices = {
            int(parts[2])
            for key in state_dict
            if key.startswith("encoder.layers.")
            for parts in [key.split(".")]
            if len(parts) > 3 and parts[2].isdigit()
        }
        decoder_layer_indices = {
            int(parts[2])
            for key in state_dict
            if key.startswith("decoder.layers.")
            for parts in [key.split(".")]
            if len(parts) > 3 and parts[2].isdigit()
        }
        model_config = {
            "audio_dim": int(state_dict["context_encoder.audio_feature_map.weight"].shape[1]),
            "feature_dim": int(state_dict["context_encoder.audio_feature_map.weight"].shape[0]),
            "output_dim": int(state_dict["output_head.3.weight"].shape[0]),
            "num_encoder_layers": (max(encoder_layer_indices) + 1 if encoder_layer_indices else 6),
            "num_decoder_layers": (max(decoder_layer_indices) + 1 if decoder_layer_indices else 6),
        }
        model = DyadicContinuousTransformer(
            audio_dim=int(model_config["audio_dim"]),
            output_dim=int(model_config["output_dim"]),
            feature_dim=int(model_config["feature_dim"]),
            n_heads=args.dyadic_n_heads,
            num_encoder_layers=int(model_config["num_encoder_layers"]),
            num_decoder_layers=int(model_config["num_decoder_layers"]),
            dropout=args.dyadic_dropout,
            video_chunk_size=args.dyadic_video_chunk_size,
        ).to(device)
    elif checkpoint_family == "listenformer":
        model_config = _infer_listenformer_config(state_dict)
        model = LookingFaceListenFormer(
            audio_dim=int(model_config["audio_dim"]),
            output_dim=int(model_config["output_dim"]),
            feature_dim=int(model_config["feature_dim"]),
            n_heads=args.listenformer_n_heads,
            num_encoder_layers=int(model_config["num_encoder_layers"]),
            num_decoder_layers=int(model_config["num_decoder_layers"]),
            dropout=args.listenformer_dropout,
            video_chunk_size=args.listenformer_video_chunk_size,
        ).to(device)
    elif checkpoint_family == "dualtalk":
        dualtalk_config = _resolve_dualtalk_config(ckpt, state_dict)
        model_config = dict(dualtalk_config)
        model = LookingFaceDualTalk(
            audio_dim=int(dualtalk_config["audio_dim"]),
            output_dim=int(dualtalk_config["output_dim"]),
            feature_dim=int(dualtalk_config["feature_dim"]),
            n_heads=int(dualtalk_config.get("n_heads", args.dualtalk_n_heads)),
            interaction_layers=int(dualtalk_config["interaction_layers"]),
            decoder_layers=int(dualtalk_config["decoder_layers"]),
            dropout=float(dualtalk_config.get("dropout", args.dualtalk_dropout)),
            video_chunk_size=int(dualtalk_config.get("video_chunk_size", args.dualtalk_video_chunk_size)),
            modulation_factor=float(dualtalk_config.get("modulation_factor", args.dualtalk_modulation_factor)),
        ).to(device)
    elif checkpoint_family == "regnn":
        regnn_config = _infer_regnn_config(state_dict)
        model_config = dict(regnn_config)
        model_config["output_dim"] = int(regnn_config["target_dim"])
        model = LookingFaceREGNN(
            audio_dim=int(regnn_config["audio_dim"]),
            target_dim=int(regnn_config["target_dim"]),
            fused_dim=int(regnn_config["fused_dim"]),
            num_frames=int(regnn_config["num_frames"]),
            edge_dim=int(regnn_config["edge_dim"]),
            neighbors=args.regnn_neighbors,
            layers=int(regnn_config["layers"]),
            act_type=args.regnn_act_type,
            noise_threshold=(args.regnn_noise_threshold if args.regnn_noise_threshold > 0 else None),
        ).to(device)
    else:
        raise RuntimeError(
            f"Expected a motion diffusion, motion flow matching, motion TransVAE, Dyadic DIM, ListenFormer, DualTalk, or REGNN checkpoint, got {checkpoint_family!r}"
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_config, checkpoint_family


def export_sequence(
    seq_id: str,
    entry: dict[str, str],
    model: torch.nn.Module,
    checkpoint_family: str,
    output_dim: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    with np.load(entry["flame_npz"]) as label_npz:
        n_frames = int(label_npz["expr"].shape[0])
        audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(
            seq_id,
            n_frames,
            args.video_canvas_size,
        )
        audio_t = audio_t.to(device)
        video_t = video_t.to(device)
        lengths = lengths.to(device)
        padding_mask = padding_mask.to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if checkpoint_family == "motion_diffusion":
                    prediction = model.sample(
                        left_audio_feat=audio_t,
                        left_video_frames=video_t,
                        padding_mask=padding_mask,
                    )
                elif checkpoint_family == "motion_flow_matching":
                    prediction = model.sample(
                        left_audio_feat=audio_t,
                        left_video_frames=video_t,
                        padding_mask=padding_mask,
                    )
                elif checkpoint_family == "motion_transvae":
                    prediction, _ = model(
                        audio_t,
                        video_t,
                        lengths=lengths,
                        padding_mask=padding_mask,
                    )
                elif checkpoint_family == "dyadic_dim":
                    prediction, _ = model(
                        left_audio_feat=audio_t,
                        left_video_frames=video_t,
                        lengths=lengths,
                        padding_mask=padding_mask,
                        target=None,
                    )
                elif checkpoint_family == "listenformer":
                    prediction, _ = model(
                        left_audio_feat=audio_t,
                        left_video_frames=video_t,
                        lengths=lengths,
                        padding_mask=padding_mask,
                        target=None,
                    )
                elif checkpoint_family == "dualtalk":
                    prediction, _ = model(
                        left_audio_feat=audio_t,
                        left_video_frames=video_t,
                        lengths=lengths,
                        padding_mask=padding_mask,
                    )
                elif checkpoint_family == "regnn":
                    prediction = model.predict_sequence(
                        left_audio_feat=audio_t,
                        left_video=video_t,
                        lengths=lengths,
                    )
                else:
                    raise RuntimeError(f"Unsupported checkpoint family: {checkpoint_family!r}")

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
    model, model_config, checkpoint_family = build_model(args, checkpoint_path, device)
    output_dim = int(model_config["output_dim"])

    completed = 0
    skipped = 0
    failed = 0

    print(f"Resolved {len(seq_ids)} sequences from {split_path}")
    print(f"Loaded manifest once: {len(manifest)} sequences")
    print(f"Loaded {checkpoint_family} checkpoint once: {checkpoint_path}")
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

        print(f"[{index}/{len(seq_ids)}] Running {checkpoint_family} inference for {seq_id}")
        try:
            written_path = export_sequence(
                seq_id=seq_id,
                entry=entry,
                model=model,
                checkpoint_family=checkpoint_family,
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