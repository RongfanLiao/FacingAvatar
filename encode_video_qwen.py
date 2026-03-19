"""Encode video files into embeddings using Qwen2.5-VL-7B vision encoder."""

import glob
import os

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VIDEO_DIR = os.path.join(DATA_DIR, "video")
EMB_DIR = os.path.join(DATA_DIR, "video_embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda:0"

# ── Load model & processor ──────────────────────────────────────────────────
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

# ── Process videos ───────────────────────────────────────────────────────────
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*left*.mp4")))
print(f"Found {len(video_files)} video files to encode.\n")

for vpath in video_files:
    stem = os.path.splitext(os.path.basename(vpath))[0]
    out_path = os.path.join(EMB_DIR, f"{stem}.npy")
    if os.path.exists(out_path):
        print(f"  Skip (exists): {stem}")
        continue

    # Build a minimal message so the processor handles video frame sampling
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
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )
        # Use the last hidden state; average over all token positions that
        # correspond to video pixels (we take the mean over the full sequence
        # as a simple, effective pooling strategy).
        last_hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_dim)
        embedding = last_hidden.mean(dim=1).squeeze(0)   # (hidden_dim,)
        embedding = embedding.cpu().float().numpy()

    np.save(out_path, embedding)
    print(f"  Saved: {stem}  shape={embedding.shape}")

    # Free VRAM between videos
    del inputs, outputs, last_hidden, embedding
    torch.cuda.empty_cache()

print(f"\nEmbeddings saved to: {EMB_DIR}")
