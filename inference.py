"""
Inference script: predict FLAME parameters from audio + video embeddings.

Usage:
    python inference.py --seq_id 2920
    python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import shutil

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from config import (
    AUDIO_EMB_DIR, VIDEO_EMB_DIR,
    WHISPER_MAX_FRAMES, WHISPER_CHUNK_SEC, CKPT_DIR, DEVICE,
)
from model import AudioVisualFLAMEModel
from dataset import _get_audio_duration, _interpolate_features, FLAME_KEYS
from manifest import load_manifest


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
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
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


def main():
    parser = argparse.ArgumentParser(description="Predict FLAME params from audio+video")
    parser.add_argument("--seq_id", type=str, required=True, help="Sequence ID (e.g. 2920)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CKPT_DIR, "best_model.pt"))
    parser.add_argument("--n_frames", type=int, default=None, help="Number of output frames (default: auto from duration)")
    parser.add_argument("--output_dir", type=str, default="output/predicted", help="Output directory")
    parser.add_argument("--gen_vis", action="store_true", help="Pack output into a per-sequence dir with foreground_image.png and transforms.json for visualization")
    parser.add_argument("--smooth", type=int, default=0, help="Temporal smoothing window size (0=disabled)")
    args = parser.parse_args()

    # Load label npz for static keys and frame count
    manifest = load_manifest()
    entry = manifest[args.seq_id]
    label_npz = np.load(entry["flame_npz"])
    label_dir = os.path.dirname(entry["flame_npz"])

    # Use label frame count if not overridden, so all params stay aligned
    n_frames = args.n_frames or label_npz["expr"].shape[0]

    print(f"Predicting FLAME params for sequence {args.seq_id} ({n_frames} frames)...")
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
