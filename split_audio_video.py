"""Split audio and video from MP4 files containing 'left' in their filenames."""

import glob
import os
import subprocess

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
VIDEO_DIR = os.path.join(DATA_DIR, "video")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

mp4_files = sorted(glob.glob(os.path.join(DATA_DIR, "*left*.mp4")))
print(f"Found {len(mp4_files)} MP4 files with 'left' in the name.")

for mp4 in mp4_files:
    stem = os.path.splitext(os.path.basename(mp4))[0]
    audio_out = os.path.join(AUDIO_DIR, f"{stem}.wav")
    video_out = os.path.join(VIDEO_DIR, f"{stem}.mp4")

    # Extract audio (WAV, no re-encode for lossless)
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
