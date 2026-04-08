"""Split audio and video from LookingFace MP4 files."""

import argparse
import os
import subprocess

from config import AUDIO_DIR, VIDEO_DIR
from manifest import load_manifest


def split_audio_video(
    extract_audio: bool = True,
    extract_video: bool = True,
    rebuild_manifest: bool = True,
    continue_on_error: bool = True,
):
    """Extract audio WAVs and/or video-only MP4s from all LookingFace left videos."""
    if extract_audio:
        os.makedirs(AUDIO_DIR, exist_ok=True)
    if extract_video:
        os.makedirs(VIDEO_DIR, exist_ok=True)

    manifest = load_manifest(rebuild=rebuild_manifest)
    mp4_files = sorted(
        [(seq_id, entry["left_mp4"]) for seq_id, entry in manifest.items()]
    )
    print(f"Found {len(mp4_files)} MP4 files to process.")
    failures: list[tuple[str, str]] = []

    for seq_id, mp4 in mp4_files:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        audio_out = os.path.join(AUDIO_DIR, f"{stem}.wav")
        video_out = os.path.join(VIDEO_DIR, f"{stem}.mp4")

        audio_ready = (not extract_audio) or os.path.exists(audio_out)
        video_ready = (not extract_video) or os.path.exists(video_out)
        if audio_ready and video_ready:
            print(f"  Skip (exists): {stem}")
            continue

        try:
            if extract_audio and not os.path.exists(audio_out):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp4, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_out],
                    check=True,
                    capture_output=True,
                )

            if extract_video and not os.path.exists(video_out):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp4, "-an", "-c:v", "copy", video_out],
                    check=True,
                    capture_output=True,
                )

            print(f"  Done: {stem}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
            failures.append((stem, stderr))
            print(f"  ERROR: {stem}")
            if stderr:
                print(f"    {stderr.splitlines()[-1]}")
            if not continue_on_error:
                raise

    if extract_audio:
        print(f"\nAudio saved to: {AUDIO_DIR}")
    if extract_video:
        print(f"Video saved to: {VIDEO_DIR}")
    if failures:
        print(f"\nCompleted with {len(failures)} failures:")
        for stem, message in failures[:20]:
            print(f"  {stem}: {message.splitlines()[-1] if message else 'ffmpeg failed'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio WAVs and/or video-only MP4s from LookingFace left videos")
    parser.add_argument("--audio-only", action="store_true", help="Extract only audio WAVs")
    parser.add_argument("--video-only", action="store_true", help="Extract only video-only MP4s")
    parser.add_argument("--no-rebuild-manifest", action="store_true", help="Use cached manifest without rescanning")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first ffmpeg error")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_audio = True
    extract_video = True
    if args.audio_only:
        extract_video = False
    if args.video_only:
        extract_audio = False
    split_audio_video(
        extract_audio=extract_audio,
        extract_video=extract_video,
        rebuild_manifest=not args.no_rebuild_manifest,
        continue_on_error=not args.fail_fast,
    )
