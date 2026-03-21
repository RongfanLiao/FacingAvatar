"""
LookingFace dataset manifest: discovers and caches the mapping from
sequence IDs to their source video and FLAME annotation paths.

The manifest is a JSON file at data/manifest.json mapping:
  seq_id -> {"left_mp4": "/abs/path/to/_left.mp4", "flame_npz": "/abs/path/to/flame_param.npz"}

Usage:
  from manifest import load_manifest
  manifest = load_manifest()              # loads from cache, or scans + caches
  manifest = load_manifest(rebuild=True)  # force rescan
"""

import json
import os
from pathlib import Path

from config import BASE_DIR, LOOKINGFACE_DIR

MANIFEST_PATH = os.path.join(BASE_DIR, "data", "manifest.json")


def _scan_lookingface(root: str) -> dict[str, dict[str, str]]:
    """Recursively scan LookingFace and pair _left.mp4 with flame_param.npz."""
    root = Path(root)

    # Find all _left.mp4 files
    left_mp4s = {}
    for mp4 in root.rglob("*_left.mp4"):
        seq_id = mp4.stem.replace("_left", "")
        left_mp4s[seq_id] = str(mp4)

    # Find all flame_param.npz files
    flame_npzs = {}
    for npz in root.rglob("flame_param.npz"):
        right_dir = npz.parent.name
        seq_id = right_dir.replace("_right", "")
        flame_npzs[seq_id] = str(npz)

    # Join on seq_id
    manifest = {}
    for seq_id in sorted(left_mp4s.keys()):
        if seq_id in flame_npzs:
            manifest[seq_id] = {
                "left_mp4": left_mp4s[seq_id],
                "flame_npz": flame_npzs[seq_id],
            }
        else:
            print(f"  Warning: {seq_id} has _left.mp4 but no flame_param.npz, skipping")

    unmatched = set(flame_npzs.keys()) - set(left_mp4s.keys())
    if unmatched:
        print(f"  Warning: {len(unmatched)} flame_param.npz without matching _left.mp4")

    return manifest


def load_manifest(rebuild: bool = False) -> dict[str, dict[str, str]]:
    """Load manifest from cache, or scan and cache if missing/rebuild requested."""
    if not rebuild and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
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
