"""Split audio and video from LookingFace MP4 files."""

import os
import subprocess

from config import AUDIO_DIR, VIDEO_DIR
from manifest import load_manifest


def split_audio_video():
    """Extract audio (WAV) and video-only (MP4) from all LookingFace left videos."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    manifest = load_manifest(rebuild=True)
    mp4_files = sorted(
        [(seq_id, entry["left_mp4"]) for seq_id, entry in manifest.items()]
    )
    print(f"Found {len(mp4_files)} MP4 files to process.")

    for seq_id, mp4 in mp4_files:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        audio_out = os.path.join(AUDIO_DIR, f"{stem}.wav")
        video_out = os.path.join(VIDEO_DIR, f"{stem}.mp4")

        if os.path.exists(audio_out) and os.path.exists(video_out):
            print(f"  Skip (exists): {stem}")
            continue

        # Extract audio (WAV, 16 kHz, lossless)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_out],
            check=True,
            capture_output=True,
        )

        # Extract video (no audio, copy video codec)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4, "-an", "-c:v", "copy", video_out],
            check=True,
            capture_output=True,
        )

        print(f"  Done: {stem}")

    print(f"\nAudio saved to: {AUDIO_DIR}")
    print(f"Video saved to: {VIDEO_DIR}")


if __name__ == "__main__":
    split_audio_video()
