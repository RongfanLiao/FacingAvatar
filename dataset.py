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


class AudioVisualFLAMEDataset(Dataset):
    """
    Dataset that loads temporally-aligned audio features, video embeddings,
    and per-frame FLAME parameter labels.
    """

    def __init__(self, seq_ids: list[str]):
        self.samples = []
        for sid in seq_ids:
            audio_path = os.path.join(AUDIO_EMB_DIR, f"{sid}_left_whisper.npy")
            video_path = os.path.join(VIDEO_EMB_DIR, f"{sid}_left.npy")
            flame_path = _find_flame_npz(sid)
            if flame_path is None:
                continue
            self.samples.append({
                "seq_id": sid,
                "audio_path": audio_path,
                "video_path": video_path,
                "flame_path": flame_path,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Load raw embeddings
        audio_emb = np.load(s["audio_path"])   # (1500, 1280)
        video_emb = np.load(s["video_path"])   # (3584,)

        # Load FLAME labels
        npz = np.load(s["flame_path"])
        flame = {k: npz[k] for k in FLAME_KEYS}
        n_frames = flame["expr"].shape[0]

        # Temporal alignment: crop valid audio frames, interpolate to FLAME fps
        duration = _get_audio_duration(s["seq_id"])
        valid_frames = int(duration / WHISPER_CHUNK_SEC * WHISPER_MAX_FRAMES)
        valid_frames = min(valid_frames, audio_emb.shape[0])
        audio_cropped = audio_emb[:valid_frames]
        audio_aligned = _interpolate_features(audio_cropped, n_frames)

        # Concatenate FLAME params into single target
        flame_target = np.concatenate(
            [flame[k] for k in FLAME_KEYS], axis=-1
        )

        return {
            "seq_id": s["seq_id"],
            "audio_feat": torch.from_numpy(audio_aligned).float(),
            "video_feat": torch.from_numpy(video_emb).float(),
            "flame_target": torch.from_numpy(flame_target).float(),
            "n_frames": n_frames,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and create padding masks."""
    audio_feats = [b["audio_feat"] for b in batch]
    video_feats = torch.stack([b["video_feat"] for b in batch])
    flame_targets = [b["flame_target"] for b in batch]
    lengths = torch.tensor([b["n_frames"] for b in batch])
    seq_ids = [b["seq_id"] for b in batch]

    audio_padded = pad_sequence(audio_feats, batch_first=True)
    flame_padded = pad_sequence(flame_targets, batch_first=True)

    max_len = audio_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    return {
        "seq_ids": seq_ids,
        "audio_feat": audio_padded,
        "video_feat": video_feats,
        "flame_target": flame_padded,
        "padding_mask": mask,
        "lengths": lengths,
    }


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    all_seqs = discover_sequences()
    if CONVERGENCE_SEQ_IDS:
        seq_set = set(CONVERGENCE_SEQ_IDS)
        all_seqs = [s for s in all_seqs if s in seq_set]

    if TRAIN_VAL_SAME_SEQS:
        train_seqs = list(all_seqs)
        val_seqs = list(all_seqs)
    else:
        # Reproducible 70/30 random split
        shuffled = list(all_seqs)
        random.Random(SPLIT_SEED).shuffle(shuffled)
        n_train = int(len(shuffled) * TRAIN_RATIO)
        train_seqs = sorted(shuffled[:n_train])
        val_seqs = sorted(shuffled[n_train:])

    print(f"Sequences — total: {len(all_seqs)}, train: {len(train_seqs)}, val: {len(val_seqs)}")
    if CONVERGENCE_SEQ_IDS:
        print(f"  Filtered to CONVERGENCE_SEQ_IDS: {CONVERGENCE_SEQ_IDS}")
    if TRAIN_VAL_SAME_SEQS:
        print("  Mode: convergence test (train == val)")

    # Export split record
    split_path = os.path.join(DATA_DIR, "split.json")
    split_record = {"train": train_seqs, "val": val_seqs}
    with open(split_path, "w") as f:
        json.dump(split_record, f, indent=2)
    print(f"  Split record saved to: {split_path}")

    train_ds = AudioVisualFLAMEDataset(train_seqs)
    val_ds = AudioVisualFLAMEDataset(val_seqs)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
