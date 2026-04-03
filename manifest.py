"""
LookingFace dataset manifest: discovers and caches the mapping from sequence
IDs to paired source videos and right-side FLAME annotations.

The manifest is a JSON file at data/manifest.json mapping:
    seq_id -> {
            "left_mp4": "/abs/path/to/_left.mp4",
            "right_mp4": "/abs/path/to/_right.mp4",
            "flame_npz": "/abs/path/to/flame_param.npz",
            "flame_dir": "/abs/path/to/_right"
    }

Usage:
    from manifest import load_manifest
    manifest = load_manifest()              # loads from cache, or scans + caches
    manifest = load_manifest(rebuild=True)  # force rescan
"""

import json
import os
from pathlib import Path

from config import BASE_DIR, DOCUMENTARY_DIR, LOOKINGFACE_DIR

MANIFEST_PATH = os.path.join(BASE_DIR, "data", "manifest.json")
DOCUMENTARY_MANIFEST_PATH = os.path.join(BASE_DIR, "data", "documentary_manifest.json")


def _infer_right_mp4(entry: dict[str, str]) -> str | None:
    """Infer a right-side MP4 path for older cached manifest entries."""
    right_mp4 = entry.get("right_mp4")
    if right_mp4:
        return right_mp4

    left_mp4 = entry.get("left_mp4")
    if left_mp4:
        candidate = str(Path(left_mp4).with_name(Path(left_mp4).name.replace("_left.mp4", "_right.mp4")))
        if candidate != left_mp4 and os.path.exists(candidate):
            return candidate

    flame_npz = entry.get("flame_npz")
    if flame_npz:
        flame_dir = Path(flame_npz).parent
        seq_id = flame_dir.name.removesuffix("_right")
        candidate = flame_dir.parent / f"{seq_id}_right.mp4"
        if candidate.exists():
            return str(candidate)

    return None


def _normalize_manifest_entry(entry: dict[str, str]) -> dict[str, str]:
    """Upgrade old manifest entries to the paired sample schema."""
    normalized = dict(entry)
    flame_npz = normalized.get("flame_npz")
    if flame_npz and "flame_dir" not in normalized:
        normalized["flame_dir"] = str(Path(flame_npz).parent)

    right_mp4 = _infer_right_mp4(normalized)
    if right_mp4 is not None:
        normalized["right_mp4"] = right_mp4

    return normalized


def _scan_lookingface(root: str) -> dict[str, dict[str, str]]:
    """Recursively scan LookingFace and pair left/right videos with FLAME data."""
    root = Path(root)

    # Find all _left.mp4 files
    left_mp4s = {}
    for mp4 in root.rglob("*_left.mp4"):
        seq_id = mp4.stem.removesuffix("_left")
        left_mp4s[seq_id] = str(mp4)

    # Find all _right.mp4 files
    right_mp4s = {}
    for mp4 in root.rglob("*_right.mp4"):
        seq_id = mp4.stem.removesuffix("_right")
        right_mp4s[seq_id] = str(mp4)

    # Find all flame_param.npz files
    flame_npzs = {}
    for npz in root.rglob("flame_param.npz"):
        right_dir = npz.parent.name
        seq_id = right_dir.removesuffix("_right")
        flame_npzs[seq_id] = str(npz)

    # Join on seq_id
    manifest = {}
    for seq_id in sorted(left_mp4s.keys()):
        if seq_id in flame_npzs:
            manifest[seq_id] = {
                "left_mp4": left_mp4s[seq_id],
                "right_mp4": right_mp4s.get(seq_id),
                "flame_npz": flame_npzs[seq_id],
                "flame_dir": str(Path(flame_npzs[seq_id]).parent),
            }
        else:
            print(f"  Warning: {seq_id} has _left.mp4 but no flame_param.npz, skipping")

    unmatched = set(flame_npzs.keys()) - set(left_mp4s.keys())
    if unmatched:
        print(f"  Warning: {len(unmatched)} flame_param.npz without matching _left.mp4")

    missing_right = [seq_id for seq_id in manifest if not manifest[seq_id].get("right_mp4")]
    if missing_right:
        print(f"  Warning: {len(missing_right)} paired samples missing _right.mp4")

    return manifest


def _scan_documentary(root: str) -> dict[str, dict[str, str]]:
    """Recursively scan a documentary data directory with nested video_title/person_id layout."""
    root = Path(root)

    left_mp4s = {}
    for mp4 in root.rglob("*_left.mp4"):
        seq_id = mp4.stem.removesuffix("_left")
        left_mp4s[seq_id] = str(mp4)

    right_mp4s = {}
    for mp4 in root.rglob("*_right.mp4"):
        seq_id = mp4.stem.removesuffix("_right")
        right_mp4s[seq_id] = str(mp4)

    flame_npzs = {}
    for npz in root.rglob("flame_param.npz"):
        right_dir = npz.parent.name
        seq_id = right_dir.removesuffix("_right")
        flame_npzs[seq_id] = str(npz)

    manifest = {}
    for seq_id in sorted(left_mp4s.keys()):
        if seq_id in flame_npzs:
            manifest[seq_id] = {
                "left_mp4": left_mp4s[seq_id],
                "right_mp4": right_mp4s.get(seq_id),
                "flame_npz": flame_npzs[seq_id],
                "flame_dir": str(Path(flame_npzs[seq_id]).parent),
            }
        else:
            print(f"  Warning: {seq_id} has _left.mp4 but no flame_param.npz, skipping")

    unmatched = set(flame_npzs.keys()) - set(left_mp4s.keys())
    if unmatched:
        print(f"  Warning: {len(unmatched)} flame_param.npz without matching _left.mp4")

    missing_right = [seq_id for seq_id in manifest if not manifest[seq_id].get("right_mp4")]
    if missing_right:
        print(f"  Warning: {len(missing_right)} paired samples missing _right.mp4")

    return manifest


def load_documentary_manifest(rebuild: bool = False) -> dict[str, dict[str, str]]:
    """Load documentary manifest from cache, or scan and cache if missing/rebuild requested."""
    if not rebuild and os.path.exists(DOCUMENTARY_MANIFEST_PATH):
        with open(DOCUMENTARY_MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        normalized = {
            seq_id: _normalize_manifest_entry(entry)
            for seq_id, entry in manifest.items()
        }
        if normalized != manifest:
            with open(DOCUMENTARY_MANIFEST_PATH, "w") as f:
                json.dump(normalized, f, indent=2, sort_keys=True)
            manifest = normalized
        print(f"Loaded documentary manifest: {len(manifest)} sequences from {DOCUMENTARY_MANIFEST_PATH}")
        return manifest

    print(f"Scanning {DOCUMENTARY_DIR} for sequences...")
    manifest = _scan_documentary(DOCUMENTARY_DIR)

    os.makedirs(os.path.dirname(DOCUMENTARY_MANIFEST_PATH), exist_ok=True)
    with open(DOCUMENTARY_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Documentary manifest cached: {len(manifest)} sequences -> {DOCUMENTARY_MANIFEST_PATH}")

    return manifest


def load_manifest(rebuild: bool = False) -> dict[str, dict[str, str]]:
    """Load manifest from cache, or scan and cache if missing/rebuild requested."""
    if not rebuild and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        normalized = {
            seq_id: _normalize_manifest_entry(entry)
            for seq_id, entry in manifest.items()
        }
        if normalized != manifest:
            os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
            with open(MANIFEST_PATH, "w") as f:
                json.dump(normalized, f, indent=2, sort_keys=True)
            manifest = normalized
        print(f"Loaded manifest: {len(manifest)} sequences from {MANIFEST_PATH}")
        return manifest

    print(f"Scanning {LOOKINGFACE_DIR} for sequences...")
    manifest = _scan_lookingface(LOOKINGFACE_DIR)

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Manifest cached: {len(manifest)} sequences -> {MANIFEST_PATH}")

    return manifest


if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    load_manifest(rebuild=rebuild)
