"""
Dataset for Audio-Visual FLAME parameter prediction.

Handles:
- Loading pre-extracted Whisper audio embeddings and Qwen video embeddings
- Loading tracked FLAME parameter labels from .npz files
- Temporal alignment: cropping valid Whisper frames and interpolating to FLAME fps
- Collation with padding and masks for variable-length sequences
"""

import json
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import random

from config import (
    AUDIO_DIR, AUDIO_EMB_DIR, VIDEO_EMB_DIR, DATA_DIR,
    WHISPER_MAX_FRAMES, WHISPER_CHUNK_SEC, AUDIO_SR,
    TRAIN_RATIO, SPLIT_SEED, TRAIN_VAL_SAME_SEQS,
    CONVERGENCE_SEQ_IDS, BATCH_SIZE, NUM_WORKERS,
)
from manifest import load_manifest


# ── FLAME parameter keys we predict (per-frame, identity-agnostic) ───────────
FLAME_KEYS = ["expr", "jaw_pose", "rotation", "neck_pose", "eyes_pose", "translation"]

# ── Module-level manifest cache ──────────────────────────────────────────────
_MANIFEST = None


def _get_manifest() -> dict[str, dict[str, str]]:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest()
    return _MANIFEST


def _get_audio_duration(seq_id: str) -> float:
    """Get duration in seconds of the WAV file for a sequence."""
    wav_path = os.path.join(AUDIO_DIR, f"{seq_id}_left.wav")
    y, sr = librosa.load(wav_path, sr=AUDIO_SR)
    return len(y) / sr


def _interpolate_features(feat: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate feature array from (T_src, D) to (target_len, D)."""
    if feat.shape[0] == target_len:
        return feat
    src_len, dim = feat.shape
    src_idx = np.linspace(0, src_len - 1, target_len)
    src_floor = np.floor(src_idx).astype(int)
    src_ceil = np.minimum(src_floor + 1, src_len - 1)
    weight = (src_idx - src_floor)[:, None]
    return feat[src_floor] * (1 - weight) + feat[src_ceil] * weight


def _find_flame_npz(seq_id: str) -> str | None:
    """Find the FLAME param .npz for a sequence ID from the manifest."""
    entry = _get_manifest().get(seq_id)
    return entry["flame_npz"] if entry else None


def discover_sequences() -> list[str]:
    """Discover all sequence IDs that have audio embeddings, video embeddings, and FLAME data."""
    manifest = _get_manifest()

    valid = []
    for seq_id in sorted(manifest.keys()):
        audio_emb = os.path.join(AUDIO_EMB_DIR, f"{seq_id}_left_whisper.npy")
        video_emb = os.path.join(VIDEO_EMB_DIR, f"{seq_id}_left.npy")
        if os.path.exists(audio_emb) and os.path.exists(video_emb):
            valid.append(seq_id)

    return valid

