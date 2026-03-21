"""
Full pipeline: preprocess LookingFace dataset and train.

Stages:
  1. Build manifest (scan LookingFace, cache to data/manifest.json)
  2. Split audio/video from raw MP4s (skip existing)
  3. Encode audio with Whisper (skip existing)
  4. Encode video with Qwen (skip existing)
  5. Train model

Usage:
    python run.py
"""


def main():
    # Stage 1 + 2: Build manifest and split audio/video
    print("=" * 60)
    print("Stage 1-2: Splitting audio/video from LookingFace MP4s")
    print("=" * 60)
    from split_audio_video import split_audio_video
    split_audio_video()

    # Stage 3: Encode audio with Whisper
    print("\n" + "=" * 60)
    print("Stage 3: Encoding audio with Whisper")
    print("=" * 60)
    from encode_audio_whisper import encode_audio_whisper
    encode_audio_whisper()

    # Stage 4: Encode video with Qwen
    print("\n" + "=" * 60)
    print("Stage 4: Encoding video with Qwen")
    print("=" * 60)
    from encode_video_qwen import encode_video_qwen
    encode_video_qwen()

    # Stage 5: Train
    print("\n" + "=" * 60)
    print("Stage 5: Training")
    print("=" * 60)
    from train import train
    train()


if __name__ == "__main__":
    main()
