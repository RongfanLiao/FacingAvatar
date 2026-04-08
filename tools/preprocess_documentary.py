"""Pre-extract wav2vec2 audio features from documentary MP4 files.

Steps:
  1. Scan data/documentary/ for *_left.mp4
  2. Extract audio via FFmpeg → data/audio/{seq_id}_left.wav (16 kHz)
  3. Extract wav2vec2 features → data/wav2vec_embeddings/{seq_id}_left.npy
  4. Build documentary manifest → data/documentary_manifest.json

Usage:
    python scripts/preprocess_documentary.py [--wav2vec_model facebook/wav2vec2-base-960h]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_DIR, AUDIO_SR, DOCUMENTARY_DIR, VIDEO_FPS, WAV2VEC_EMB_DIR
from manifest import load_documentary_manifest


def extract_audio_from_mp4(mp4_path: str, wav_path: str, sample_rate: int = 16000) -> bool:
    """Extract audio from MP4 using FFmpeg, resample to target rate."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", mp4_path, "-ar", str(sample_rate), "-ac", "1", wav_path],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  FFmpeg failed for {mp4_path}: {result.stderr.decode()}")
        return False
    return True


def extract_wav2vec_features(
    wav_path: str,
    model: Wav2Vec2Model,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    fps: int = 25,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract wav2vec2 features from a WAV file, interpolated to video FPS."""
    speech_array, sr = librosa.load(wav_path, sr=sample_rate)
    seq_len = math.ceil(len(speech_array) / sample_rate * fps)

    inputs = feature_extractor(speech_array, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # (1, T_enc, 768)

    # Interpolate to match video frame count
    # F.interpolate expects (B, C, T) for 1D interpolation
    features = hidden_states.transpose(1, 2)  # (1, 768, T_enc)
    features = F.interpolate(features, size=seq_len, mode="linear", align_corners=True)
    features = features.transpose(1, 2).squeeze(0)  # (seq_len, 768)

    return features.cpu().float().numpy()


def scan_left_mp4s(root: str) -> list[tuple[str, str]]:
    """Find all *_left.mp4 files, return (seq_id, abs_path) pairs."""
    results = []
    for mp4 in Path(root).rglob("*_left.mp4"):
        seq_id = mp4.stem.replace("_left", "")
        results.append((seq_id, str(mp4)))
    return sorted(results, key=lambda x: x[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract wav2vec2 features from documentary data")
    parser.add_argument("--wav2vec_model", default="facebook/wav2vec2-base-960h",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load wav2vec2 model
    print(f"Loading wav2vec2 model: {args.wav2vec_model}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model)
    model = Wav2Vec2Model.from_pretrained(args.wav2vec_model).to(device)
    model.eval()
    model.feature_extractor._freeze_parameters()

    # Scan for left MP4s
    left_mp4s = scan_left_mp4s(DOCUMENTARY_DIR)
    print(f"Found {len(left_mp4s)} left MP4 files in {DOCUMENTARY_DIR}")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(WAV2VEC_EMB_DIR, exist_ok=True)

    succeeded = 0
    skipped = 0
    failed = 0

    for i, (seq_id, mp4_path) in enumerate(left_mp4s):
        wav_path = os.path.join(AUDIO_DIR, f"{seq_id}_left.wav")
        emb_path = os.path.join(WAV2VEC_EMB_DIR, f"{seq_id}_left.npy")

        if args.skip_existing and os.path.exists(emb_path):
            skipped += 1
            continue

        print(f"[{i + 1}/{len(left_mp4s)}] {seq_id}")

        # Step 1: Extract audio
        if not os.path.exists(wav_path):
            if not extract_audio_from_mp4(mp4_path, wav_path, sample_rate=AUDIO_SR):
                failed += 1
                continue

        # Step 2: Extract wav2vec2 features
        try:
            features = extract_wav2vec_features(
                wav_path, model, feature_extractor, device,
                fps=VIDEO_FPS, sample_rate=AUDIO_SR,
            )
            np.save(emb_path, features)
            print(f"  -> {emb_path}  shape={features.shape}")
            succeeded += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {skipped} skipped, {failed} failed")

    # Step 3: Build manifest
    print("\nBuilding documentary manifest...")
    manifest = load_documentary_manifest(rebuild=True)
    print(f"Manifest has {len(manifest)} sequences")


if __name__ == "__main__":
    main()
