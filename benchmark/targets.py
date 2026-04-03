"""Target conversion helpers for LookingFace benchmark adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FLAME_118_KEYS = ["expr", "jaw_pose", "rotation", "neck_pose", "eyes_pose", "translation"]
FLAME_118_DIMENSIONS = {
    "expr": 100,
    "jaw_pose": 3,
    "rotation": 3,
    "neck_pose": 3,
    "eyes_pose": 6,
    "translation": 3,
}
FLAME_118_DIM = sum(FLAME_118_DIMENSIONS.values())

FLAME_CONTENT_KEYS = ["expr", "jaw_pose", "neck_pose", "eyes_pose"]
FLAME_CONTENT_DIMENSIONS = {
    "expr": 100,
    "jaw_pose": 3,
    "neck_pose": 3,
    "eyes_pose": 6,
}
FLAME_CONTENT_DIM = sum(FLAME_CONTENT_DIMENSIONS.values())

# REACT-style reduced motion target: 52 expression + 3 rotation + 3 translation.
FLAME_58_KEYS = ["expr_52", "rotation", "translation"]
FLAME_58_DIM = 58


def _to_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def flame_dict_to_118(flame: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the native 118-d FLAME target in repo order."""
    return np.concatenate([_to_float32(flame[key]) for key in FLAME_118_KEYS], axis=-1)


def flame_118_to_motion58(flame_target: np.ndarray) -> np.ndarray:
    """Reduce native 118-d FLAME to the 58-d REACT-style motion target."""
    flame_target = _to_float32(flame_target)
    if flame_target.shape[-1] != FLAME_118_DIM:
        raise ValueError(f"Expected last dim {FLAME_118_DIM}, got {flame_target.shape[-1]}")

    expr_52 = flame_target[..., :52]
    rotation = flame_target[..., 103:106]
    translation = flame_target[..., 115:118]
    return np.concatenate([expr_52, rotation, translation], axis=-1)


def flame_dict_to_motion58(flame: dict[str, np.ndarray]) -> np.ndarray:
    """Reduce a FLAME dict directly to the 58-d REACT-style motion target."""
    expr_52 = _to_float32(flame["expr"])[..., :52]
    rotation = _to_float32(flame["rotation"])
    translation = _to_float32(flame["translation"])
    return np.concatenate([expr_52, rotation, translation], axis=-1)


def flame_118_to_content(flame_target: np.ndarray) -> np.ndarray:
    """Reduce native 118-d FLAME to expr+jaw+neck+eyes content target."""
    flame_target = _to_float32(flame_target)
    if flame_target.shape[-1] != FLAME_118_DIM:
        raise ValueError(f"Expected last dim {FLAME_118_DIM}, got {flame_target.shape[-1]}")

    expr = flame_target[..., :100]
    jaw = flame_target[..., 100:103]
    neck = flame_target[..., 106:109]
    eyes = flame_target[..., 109:115]
    return np.concatenate([expr, jaw, neck, eyes], axis=-1)


def flame_dict_to_content(flame: dict[str, np.ndarray]) -> np.ndarray:
    """Reduce a FLAME dict directly to expr+jaw+neck+eyes content target."""
    return np.concatenate([
        _to_float32(flame["expr"]),
        _to_float32(flame["jaw_pose"]),
        _to_float32(flame["neck_pose"]),
        _to_float32(flame["eyes_pose"]),
    ], axis=-1)


def flame_npz_to_targets(flame_npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load native and reduced targets from a flame_param.npz file."""
    with np.load(flame_npz_path) as npz:
        flame = {key: _to_float32(npz[key]) for key in FLAME_118_KEYS}

    target_118 = flame_dict_to_118(flame)
    target_58 = flame_118_to_motion58(target_118)
    target_content = flame_118_to_content(target_118)
    return {
        "flame_dict": flame,
        "flame_target_118": target_118,
        "flame_target_58": target_58,
        "flame_target_content": target_content,
    }