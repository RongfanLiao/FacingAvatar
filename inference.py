"""Inference entrypoint for both legacy and motion TransVAE checkpoints.

Usage:
    python inference.py --seq_id 2920
    python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/motion_transvae_lookingface/best.pt
"""

import argparse
import os
import shutil

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.nn.utils.rnn import pad_sequence

from benchmark.lookingface import _fit_frame_to_canvas
from benchmark.motion_transvae import MotionOnlyTransformerVAE
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM
from config import (
    AUDIO_EMB_DIR, VIDEO_EMB_DIR,
    WAV2VEC_EMB_DIR, WHISPER_MAX_FRAMES, WHISPER_CHUNK_SEC, CKPT_DIR, DEVICE,
)
from model import AudioVisualFLAMEModel
from dataset import _get_audio_duration, _interpolate_features, FLAME_KEYS
from manifest import load_manifest
from decord import VideoReader, cpu as decord_cpu


def _load_checkpoint(checkpoint: str, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def _detect_checkpoint_family(checkpoint_state: dict[str, torch.Tensor]) -> str:
    if "decoder.output_head.weight" in checkpoint_state:
        return "motion_transvae"
    if "flame_head.expr_head.weight" in checkpoint_state:
        return "legacy_av"
    raise RuntimeError("Unsupported checkpoint format")


def _build_motion_transvae_inputs(seq_id: str, n_frames: int, video_canvas_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = load_manifest()
    entry = manifest.get(seq_id)
    if entry is None:
        raise KeyError(f"Sequence {seq_id} not found in manifest")

    wav2vec_path = os.path.join(WAV2VEC_EMB_DIR, f"{seq_id}_left.npy")
    if not os.path.exists(wav2vec_path):
        raise FileNotFoundError(f"Missing wav2vec feature file: {wav2vec_path}")

    wav2vec_feat = np.load(wav2vec_path)
    wav2vec_feat = _interpolate_features(wav2vec_feat, n_frames)
    audio_tensor = torch.from_numpy(np.asarray(wav2vec_feat, dtype=np.float32)).float()

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
    video_tensor = torch.stack(frames)

    lengths = torch.tensor([n_frames], dtype=torch.long)
    padding_mask = torch.zeros((1, n_frames), dtype=torch.bool)
    return audio_tensor.unsqueeze(0), video_tensor.unsqueeze(0), lengths, padding_mask


def _motion_prediction_to_flame(prediction: np.ndarray, label_npz: np.lib.npyio.NpzFile, output_dim: int) -> dict[str, np.ndarray]:
    results = {key: np.asarray(label_npz[key]) for key in label_npz.files}

    if output_dim == FLAME_CONTENT_DIM:
        results["expr"] = prediction[:, :100]
        results["jaw_pose"] = prediction[:, 100:103]
        results["neck_pose"] = prediction[:, 103:106]
        results["eyes_pose"] = prediction[:, 106:112]
        return results

    if output_dim == FLAME_58_DIM:
        expr = np.asarray(label_npz["expr"]).copy()
        expr[:, :52] = prediction[:, :52]
        results["expr"] = expr
        results["rotation"] = prediction[:, 52:55]
        results["translation"] = prediction[:, 55:58]
        return results

    raise RuntimeError(f"Unsupported motion TransVAE output dim: {output_dim}")


def predict(seq_id: str, checkpoint: str, n_frames: int | None = None) -> dict[str, np.ndarray]:
    """
    Predict FLAME parameters for a given sequence.

    Args:
        seq_id: Sequence ID (e.g. "2920")
        checkpoint: Path to model checkpoint
        n_frames: Number of output frames. If None, inferred from audio duration
                  assuming ~25 fps.
    """
    # Load embeddings
    audio_path = os.path.join(AUDIO_EMB_DIR, f"{seq_id}_left_whisper.npy")
    video_path = os.path.join(VIDEO_EMB_DIR, f"{seq_id}_left.npy")

    audio_emb = np.load(audio_path)   # (1500, 1280)
    video_emb = np.load(video_path)   # (3584,)

    # Temporal alignment
    duration = _get_audio_duration(seq_id)
    valid_frames = int(duration / WHISPER_CHUNK_SEC * WHISPER_MAX_FRAMES)
    valid_frames = min(valid_frames, audio_emb.shape[0])
    audio_cropped = audio_emb[:valid_frames]

    if n_frames is None:
        n_frames = int(duration * 25)  # assume ~25 fps

    audio_aligned = _interpolate_features(audio_cropped, n_frames)  # (n_frames, 1280)

    # Load model
    model = AudioVisualFLAMEModel().to(DEVICE)
    ckpt = _load_checkpoint(checkpoint, DEVICE)
    load_res = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = set(load_res.missing_keys)
    unexpected = set(load_res.unexpected_keys)
    allowed_missing = {
        "flame_head.trans_head.weight",
        "flame_head.trans_head.bias",
    }
    if unexpected or not missing.issubset(allowed_missing):
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing_keys={sorted(missing)} unexpected_keys={sorted(unexpected)}"
        )
    if missing:
        print(
            "Warning: checkpoint does not contain translation head weights; "
            "using initialized translation head for inference."
        )
    model.eval()

    # Predict
    audio_t = torch.from_numpy(audio_aligned).float().unsqueeze(0).to(DEVICE)  # (1, N, 1280)
    video_t = torch.from_numpy(video_emb).float().unsqueeze(0).to(DEVICE)      # (1, 3584)

    with torch.no_grad():
        preds = model(audio_t, video_t)

    # Convert to numpy
    results = {}
    for key in FLAME_KEYS:
        results[key] = preds[key].squeeze(0).cpu().numpy()  # (N, dim)

    return results


def predict_motion_transvae(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    output_dim = int(ckpt["model_state_dict"]["decoder.output_head.weight"].shape[0])
    model = MotionOnlyTransformerVAE(output_dim=output_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction, _ = model(audio_t, video_t, lengths=lengths, padding_mask=padding_mask)

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def main():
    parser = argparse.ArgumentParser(description="Predict FLAME params from audio+video")
    parser.add_argument("--seq_id", type=str, required=True, help="Sequence ID (e.g. 2920)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CKPT_DIR, "best_model.pt"))
    parser.add_argument("--n_frames", type=int, default=None, help="Number of output frames (default: auto from duration)")
    parser.add_argument("--output_dir", type=str, default="output/predicted", help="Output directory")
    parser.add_argument("--gen_vis", action="store_true", help="Pack output into a per-sequence dir with foreground_image.png and transforms.json for visualization")
    parser.add_argument("--smooth", type=int, default=0, help="Temporal smoothing window size (0=disabled)")
    parser.add_argument("--video_canvas_size", type=int, default=400, help="Canvas size for motion TransVAE raw-frame inference")
    args = parser.parse_args()

    # Load label npz for static keys and frame count
    manifest = load_manifest()
    entry = manifest[args.seq_id]
    label_npz = np.load(entry["flame_npz"])
    label_dir = os.path.dirname(entry["flame_npz"])
    checkpoint = _load_checkpoint(args.checkpoint, DEVICE)
    checkpoint_family = _detect_checkpoint_family(checkpoint["model_state_dict"])

    # Use label frame count if not overridden, so all params stay aligned
    n_frames = args.n_frames or label_npz["expr"].shape[0]

    print(f"Predicting FLAME params for sequence {args.seq_id} ({n_frames} frames)...")
    print(f"Checkpoint family: {checkpoint_family}")
    if checkpoint_family == "motion_transvae":
        results = predict_motion_transvae(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
        )
    else:
        results = predict(args.seq_id, args.checkpoint, n_frames)

    # Optional temporal smoothing on predicted params
    if args.smooth > 0:
        for key in FLAME_KEYS:
            results[key] = uniform_filter1d(results[key], size=args.smooth, axis=0)

    # Copy non-predicted keys (shape, canonical_*) from labels
    for k in label_npz.keys():
        if k not in results:
            results[k] = label_npz[k]

    if args.gen_vis:
        # Pack into a per-sequence directory matching label layout
        seq_out_dir = os.path.join(args.output_dir, f"{args.seq_id}_right")
        os.makedirs(seq_out_dir, exist_ok=True)

        out_path = os.path.join(seq_out_dir, "flame_param.npz")
        np.savez(out_path, **results)

        # Copy visualization assets from label dir
        for fname in ("foreground_image.png", "transforms.json"):
            src = os.path.join(label_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, seq_out_dir)
                print(f"  Copied {fname}")

    else:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{args.seq_id}_flame_pred.npz")
        np.savez(out_path, **results)

    print(f"Saved to {out_path}")
    for key, arr in results.items():
        print(f"  {key}: shape={arr.shape}")


if __name__ == "__main__":
    main()
