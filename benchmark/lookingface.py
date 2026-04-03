"""Shared paired-sample adapter for LookingFace benchmark ports."""

from __future__ import annotations

import json
import os
import random

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from benchmark.targets import flame_npz_to_targets
import torchvision.transforms as T
from decord import VideoReader, cpu as decord_cpu
from PIL import Image

from config import (
    AUDIO_DIR,
    AUDIO_EMB_DIR,
    AUDIO_SR,
    CONVERGENCE_SEQ_IDS,
    DATA_DIR,
    LOOKINGFACE_SPLITS_DIR,
    SPLIT_SEED,
    TRAIN_RATIO,
    TRAIN_VAL_SAME_SEQS,
    VIDEO_CANVAS_SIZE,
    VIDEO_EMB_DIR,
    WAV2VEC_EMB_DIR,
    WHISPER_CHUNK_SEC,
    WHISPER_MAX_FRAMES,
)
from manifest import load_documentary_manifest, load_manifest


_MANIFEST = None


def _get_manifest() -> dict[str, dict[str, str]]:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest()
    return _MANIFEST


def _get_audio_duration(seq_id: str) -> float:
    """Get duration in seconds of the left-side WAV file for a sequence."""
    wav_path = os.path.join(AUDIO_DIR, f"{seq_id}_left.wav")
    y, sr = librosa.load(wav_path, sr=AUDIO_SR)
    return len(y) / sr


def _interpolate_features(feat: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate feature array from (T_src, D) to (target_len, D)."""
    if feat.shape[0] == target_len:
        return feat
    src_len = feat.shape[0]
    src_idx = np.linspace(0, src_len - 1, target_len)
    src_floor = np.floor(src_idx).astype(int)
    src_ceil = np.minimum(src_floor + 1, src_len - 1)
    weight = (src_idx - src_floor)[:, None]
    return feat[src_floor] * (1 - weight) + feat[src_ceil] * weight


_VIDEO_NORMALIZE = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _fit_frame_to_canvas(frame: np.ndarray, canvas_size: int = VIDEO_CANVAS_SIZE) -> torch.Tensor:
    """Fit a video frame into a square canvas, scale by longer side, center, pad with mean gray.

    Args:
        frame: numpy array (H, W, 3) in uint8 RGB from decord.
        canvas_size: target canvas side length.

    Returns:
        Tensor (3, canvas_size, canvas_size) normalized to [-1, 1].
    """
    img = Image.fromarray(frame)
    w, h = img.size
    scale = canvas_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    # Gray canvas — 128/255 ≈ 0.502, becomes ~0 after Normalize([0.5],[0.5])
    canvas = Image.new("RGB", (canvas_size, canvas_size), (128, 128, 128))
    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    tensor = T.ToTensor()(canvas)  # (3, H, W) in [0, 1]
    return _VIDEO_NORMALIZE(tensor)  # (3, H, W) in [-1, 1]


def _make_sample_record(seq_id: str, entry: dict[str, str]) -> dict[str, str | None]:
    return {
        "seq_id": seq_id,
        "left_mp4": entry["left_mp4"],
        "right_mp4": entry.get("right_mp4"),
        "flame_npz": entry["flame_npz"],
        "flame_dir": entry.get("flame_dir") or os.path.dirname(entry["flame_npz"]),
        "left_audio_emb": os.path.join(AUDIO_EMB_DIR, f"{seq_id}_left_whisper.npy"),
        "left_wav2vec_emb": os.path.join(WAV2VEC_EMB_DIR, f"{seq_id}_left.npy"),
        "left_video_emb": os.path.join(VIDEO_EMB_DIR, f"{seq_id}_left.npy"),
        "right_video_emb": os.path.join(VIDEO_EMB_DIR, f"{seq_id}_right.npy"),
    }


def discover_benchmark_sequences(
    seq_ids: list[str] | None = None,
    require_left_audio: bool = True,
    require_left_video_embedding: bool = True,
    require_right_video_embedding: bool = False,
    require_right_mp4: bool = True,
    require_flame_target: bool = True,
    require_wav2vec_audio: bool = False,
    manifest: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """Discover sequence IDs that satisfy the requested paired benchmark contract."""
    if manifest is None:
        manifest = _get_manifest()
    candidate_ids = seq_ids if seq_ids is not None else sorted(manifest.keys())

    valid = []
    for seq_id in candidate_ids:
        entry = manifest.get(seq_id)
        if entry is None or "left_mp4" not in entry or "flame_npz" not in entry:
            continue

        sample = _make_sample_record(seq_id, entry)
        if not os.path.exists(sample["left_mp4"]):
            continue
        if require_right_mp4 and not sample["right_mp4"]:
            continue
        if require_right_mp4 and sample["right_mp4"] and not os.path.exists(sample["right_mp4"]):
            continue
        if require_flame_target and not os.path.exists(sample["flame_npz"]):
            continue
        if require_left_audio and not os.path.exists(sample["left_audio_emb"]):
            continue
        if require_wav2vec_audio and not os.path.exists(sample["left_wav2vec_emb"]):
            continue
        if require_left_video_embedding and not os.path.exists(sample["left_video_emb"]):
            continue
        if require_right_video_embedding and not os.path.exists(sample["right_video_emb"]):
            continue
        valid.append(seq_id)

    return sorted(valid)


def build_benchmark_split(
    split_path: str | None = None,
    seq_ids: list[str] | None = None,
    require_left_audio: bool = True,
    require_left_video_embedding: bool = True,
    require_right_video_embedding: bool = False,
    require_right_mp4: bool = True,
    require_flame_target: bool = True,
    require_wav2vec_audio: bool = False,
    manifest: dict[str, dict[str, str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Create a reproducible train/val split for paired LookingFace samples."""
    all_seqs = discover_benchmark_sequences(
        seq_ids=seq_ids,
        require_left_audio=require_left_audio,
        require_left_video_embedding=require_left_video_embedding,
        require_right_video_embedding=require_right_video_embedding,
        require_right_mp4=require_right_mp4,
        require_flame_target=require_flame_target,
        require_wav2vec_audio=require_wav2vec_audio,
        manifest=manifest,
    )
    if CONVERGENCE_SEQ_IDS:
        seq_set = set(CONVERGENCE_SEQ_IDS)
        all_seqs = [seq_id for seq_id in all_seqs if seq_id in seq_set]

    if TRAIN_VAL_SAME_SEQS:
        train_seqs = list(all_seqs)
        val_seqs = list(all_seqs)
    else:
        shuffled = list(all_seqs)
        random.Random(SPLIT_SEED).shuffle(shuffled)
        n_train = int(len(shuffled) * TRAIN_RATIO)
        train_seqs = sorted(shuffled[:n_train])
        val_seqs = sorted(shuffled[n_train:])

    if split_path is not None:
        with open(split_path, "w") as f:
            json.dump({"train": train_seqs, "val": val_seqs}, f, indent=2)

    return train_seqs, val_seqs


def load_predefined_splits(
    splits_dir: str | None = None,
    manifest: dict[str, dict[str, str]] | None = None,
    require_wav2vec_audio: bool = False,
    require_right_mp4: bool = True,
) -> dict[str, list[str]]:
    """Load predefined train/valid/test splits from dataset_splits directory.

    The split JSON files contain relative video paths like:
        "documentary/VideoTitle/person_id/0761_left.mp4"
    This function extracts seq_ids and filters against the manifest.

    Returns:
        dict mapping split name ("train", "valid", "test") to list of seq_ids.
    """
    if splits_dir is None:
        splits_dir = LOOKINGFACE_SPLITS_DIR

    if manifest is None:
        manifest = _get_manifest()

    result = {}
    for split_name in ("train", "valid", "test"):
        split_path = os.path.join(splits_dir, f"{split_name}.json")
        if not os.path.exists(split_path):
            continue

        with open(split_path) as f:
            paths = json.load(f)

        # Extract seq_ids from _left.mp4 paths only
        seq_ids = []
        for path in paths:
            if not path.endswith("_left.mp4"):
                continue
            stem = os.path.basename(path)[:-4]  # remove .mp4
            seq_id = stem.removesuffix("_left")
            seq_ids.append(seq_id)

        # Filter to only seq_ids present in the manifest with required data
        valid_seq_ids = discover_benchmark_sequences(
            seq_ids=seq_ids,
            require_left_audio=False,
            require_left_video_embedding=False,
            require_wav2vec_audio=require_wav2vec_audio,
            require_right_mp4=require_right_mp4,
            manifest=manifest,
        )

        result[split_name] = valid_seq_ids
        print(f"  Split '{split_name}': {len(seq_ids)} in file, {len(valid_seq_ids)} available")

    return result


class LookingFaceBenchmarkDataset(Dataset):
    """Shared paired-sample dataset for imported LookingFace benchmarks."""

    def __init__(
        self,
        seq_ids: list[str] | None = None,
        load_left_audio: bool = True,
        load_left_video_embedding: bool = True,
        load_right_video_embedding: bool = False,
        load_flame_target: bool = True,
        align_left_audio_to_flame: bool = True,
        include_motion58_target: bool = True,
        include_content_target: bool = True,
        require_right_mp4: bool = True,
        load_left_video_raw: bool = False,
        load_wav2vec_audio: bool = False,
        video_canvas_size: int = VIDEO_CANVAS_SIZE,
        manifest: dict[str, dict[str, str]] | None = None,
    ):
        if manifest is None:
            manifest = _get_manifest()
        discovered = discover_benchmark_sequences(
            seq_ids=seq_ids,
            require_left_audio=load_left_audio,
            require_left_video_embedding=load_left_video_embedding,
            require_right_video_embedding=load_right_video_embedding,
            require_right_mp4=require_right_mp4,
            require_flame_target=load_flame_target,
            require_wav2vec_audio=load_wav2vec_audio,
            manifest=manifest,
        )
        self.samples = [_make_sample_record(seq_id, manifest[seq_id]) for seq_id in discovered]
        self.load_left_audio = load_left_audio
        self.load_left_video_embedding = load_left_video_embedding
        self.load_right_video_embedding = load_right_video_embedding
        self.load_flame_target = load_flame_target
        self.align_left_audio_to_flame = align_left_audio_to_flame
        self.include_motion58_target = include_motion58_target
        self.include_content_target = include_content_target
        self.load_left_video_raw = load_left_video_raw
        self.load_wav2vec_audio = load_wav2vec_audio
        self.video_canvas_size = video_canvas_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = dict(self.samples[idx])
        item: dict[str, object] = {
            "seq_id": sample["seq_id"],
            "left_mp4": sample["left_mp4"],
            "right_mp4": sample["right_mp4"],
            "flame_dir": sample["flame_dir"],
        }

        n_frames = None
        if self.load_flame_target:
            targets = flame_npz_to_targets(sample["flame_npz"])
            flame_target_118 = targets["flame_target_118"]
            n_frames = flame_target_118.shape[0]
            item["flame_target_118"] = torch.from_numpy(flame_target_118).float()
            if self.include_motion58_target:
                item["flame_target_58"] = torch.from_numpy(targets["flame_target_58"]).float()
            if self.include_content_target:
                item["flame_target_content"] = torch.from_numpy(targets["flame_target_content"]).float()

        if self.load_left_audio:
            audio_feat = np.load(sample["left_audio_emb"])
            if self.align_left_audio_to_flame and n_frames is not None:
                duration = _get_audio_duration(sample["seq_id"])
                valid_frames = int(duration / WHISPER_CHUNK_SEC * WHISPER_MAX_FRAMES)
                valid_frames = min(valid_frames, audio_feat.shape[0])
                audio_feat = audio_feat[:valid_frames]
                audio_feat = _interpolate_features(audio_feat, n_frames)
            item["left_audio_feat"] = torch.from_numpy(np.asarray(audio_feat, dtype=np.float32)).float()

        if self.load_wav2vec_audio:
            wav2vec_feat = np.load(sample["left_wav2vec_emb"])
            if self.align_left_audio_to_flame and n_frames is not None:
                wav2vec_feat = _interpolate_features(wav2vec_feat, n_frames)
            item["left_audio_feat"] = torch.from_numpy(np.asarray(wav2vec_feat, dtype=np.float32)).float()

        if self.load_left_video_raw:
            vr = VideoReader(sample["left_mp4"], ctx=decord_cpu(0))
            total_video_frames = len(vr)
            # Select frame indices — subsample to match n_frames (FLAME count) if available
            if n_frames is not None and n_frames > 0 and total_video_frames != n_frames:
                indices = np.linspace(0, total_video_frames - 1, n_frames).astype(int).tolist()
            else:
                indices = list(range(total_video_frames))
            frames = []
            for fi in indices:
                frame_np = vr[fi].asnumpy()  # (H, W, 3) uint8 RGB
                frames.append(_fit_frame_to_canvas(frame_np, self.video_canvas_size))
            item["left_video_frames"] = torch.stack(frames)  # (T, 3, H, W)

        if self.load_left_video_embedding:
            left_video_feat = np.load(sample["left_video_emb"])
            item["left_video_feat"] = torch.from_numpy(np.asarray(left_video_feat, dtype=np.float32)).float()

        if self.load_right_video_embedding:
            right_video_feat = np.load(sample["right_video_emb"])
            item["right_video_feat"] = torch.from_numpy(np.asarray(right_video_feat, dtype=np.float32)).float()

        if n_frames is None:
            if "left_video_frames" in item:
                n_frames = int(item["left_video_frames"].shape[0])
            elif "left_audio_feat" in item:
                n_frames = int(item["left_audio_feat"].shape[0])
            else:
                n_frames = 0
        item["n_frames"] = n_frames
        return item


def collate_benchmark_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate paired LookingFace samples with optional padded temporal tensors."""
    collated: dict[str, object] = {
        "seq_ids": [item["seq_id"] for item in batch],
        "left_mp4": [item["left_mp4"] for item in batch],
        "right_mp4": [item["right_mp4"] for item in batch],
        "flame_dir": [item["flame_dir"] for item in batch],
    }

    lengths = torch.tensor([int(item["n_frames"]) for item in batch], dtype=torch.long)
    collated["lengths"] = lengths

    temporal_keys = ["left_audio_feat", "left_video_frames", "flame_target_118", "flame_target_58", "flame_target_content"]
    for key in temporal_keys:
        if key in batch[0]:
            padded = pad_sequence([item[key] for item in batch], batch_first=True)
            collated[key] = padded

    fixed_keys = ["left_video_feat", "right_video_feat"]
    for key in fixed_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    if any(key in collated for key in temporal_keys):
        max_len = max(int(item["n_frames"]) for item in batch)
        collated["padding_mask"] = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    return collated


def default_benchmark_split_path() -> str:
    """Location for shared LookingFace benchmark split records."""
    return os.path.join(DATA_DIR, "benchmark_split.json")