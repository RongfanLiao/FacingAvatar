"""Encode LookingFace left videos into embeddings using Qwen2.5-VL-7B vision encoder."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import DEVICE, VIDEO_DIR, VIDEO_EMB_DIR
from manifest import load_manifest

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def discover_left_videos() -> list[tuple[str, str]]:
    """Discover left videos from the manifest, falling back to copied videos if needed."""
    manifest = load_manifest()
    manifest_videos = []
    for seq_id, entry in sorted(manifest.items()):
        left_mp4 = entry.get("left_mp4")
        if left_mp4 and os.path.exists(left_mp4):
            manifest_videos.append((seq_id, left_mp4))
    if manifest_videos:
        return manifest_videos

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*left*.mp4")))
    return [(Path(video_path).stem.removesuffix("_left"), video_path) for video_path in video_files]


def encode_video_qwen() -> None:
    """Encode all discovered left videos with Qwen and save embeddings to disk."""
    os.makedirs(VIDEO_EMB_DIR, exist_ok=True)

    video_files = discover_left_videos()
    pending = []
    for seq_id, video_path in video_files:
        stem = f"{seq_id}_left"
        out_path = os.path.join(VIDEO_EMB_DIR, f"{stem}.npy")
        if not os.path.exists(out_path):
            pending.append((seq_id, video_path))

    print(f"Video embeddings: {len(video_files)} total, {len(video_files) - len(pending)} cached, {len(pending)} pending.")
    if not pending:
        print("All video embeddings cached — skipping Qwen model loading.")
        return

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    print(f"Encoding {len(pending)} video files...\n")
    for seq_id, video_path in pending:
        stem = f"{seq_id}_left"
        out_path = os.path.join(VIDEO_EMB_DIR, f"{stem}.npy")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{video_path}", "fps": 2.0},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embedding = last_hidden.mean(dim=1).squeeze(0).cpu().float().numpy()

        np.save(out_path, embedding)
        print(f"  Saved: {stem}  shape={embedding.shape}")

        del inputs, outputs, last_hidden, embedding
        torch.cuda.empty_cache()

    print(f"\nEmbeddings saved to: {VIDEO_EMB_DIR}")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    encode_video_qwen()
