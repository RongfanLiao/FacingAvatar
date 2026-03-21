"""Encode video files into embeddings using Qwen2.5-VL-7B vision encoder."""

import glob
import os

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import VIDEO_DIR, VIDEO_EMB_DIR, DEVICE

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def encode_video_qwen():
    """Encode all video files with Qwen and save embeddings to disk."""
    os.makedirs(VIDEO_EMB_DIR, exist_ok=True)

    # Check cache: skip model loading if all embeddings exist
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*left*.mp4")))
    pending = []
    for vpath in video_files:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(VIDEO_EMB_DIR, f"{stem}.npy")
        if not os.path.exists(out_path):
            pending.append(vpath)

    print(f"Video embeddings: {len(video_files)} total, {len(video_files) - len(pending)} cached, {len(pending)} pending.")
    if not pending:
        print("All video embeddings cached — skipping Qwen model loading.")
        return

    # Load model & processor
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

    # Process pending videos
    print(f"Encoding {len(pending)} video files...\n")

    for vpath in pending:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(VIDEO_EMB_DIR, f"{stem}.npy")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{vpath}", "fps": 2.0},
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
            embedding = last_hidden.mean(dim=1).squeeze(0)
            embedding = embedding.cpu().float().numpy()

        np.save(out_path, embedding)
        print(f"  Saved: {stem}  shape={embedding.shape}")

        del inputs, outputs, last_hidden, embedding
        torch.cuda.empty_cache()

    print(f"\nEmbeddings saved to: {VIDEO_EMB_DIR}")

    # Free GPU memory
    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    encode_video_qwen()
