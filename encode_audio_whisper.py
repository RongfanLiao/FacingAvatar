"""
Extract audio features with Whisper-Large-v3 encoder (no decoder) and project
them into the 3584-dim LLM embedding space via a trainable projector.

Pipeline:
  48 kHz audio → resample to 16 kHz → Whisper encoder → (T, 1280)
                                       → Projector    → (T, 3584)

Audio files are already 16 kHz WAV from the earlier split step.
"""

import glob
import os

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor

from config import AUDIO_DIR, AUDIO_EMB_DIR, DEVICE

MODEL_ID = "openai/whisper-large-v3"
TARGET_SR = 16000
WHISPER_DIM = 1280   # Whisper-Large-v3 encoder hidden size
LLM_DIM = 3584       # Qwen2.5-VL-7B hidden size


class AudioProjector(nn.Module):
    """Two-layer MLP that maps Whisper features into the LLM embedding space."""

    def __init__(self, input_dim: int = WHISPER_DIM, output_dim: int = LLM_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def encode_audio_whisper():
    """Encode all audio files with Whisper and save embeddings to disk."""
    os.makedirs(AUDIO_EMB_DIR, exist_ok=True)

    # Check cache: skip model loading if all embeddings exist
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*left*.wav")))
    pending = []
    for apath in audio_files:
        stem = os.path.splitext(os.path.basename(apath))[0]
        enc_path = os.path.join(AUDIO_EMB_DIR, f"{stem}_whisper.npy")
        proj_path = os.path.join(AUDIO_EMB_DIR, f"{stem}_proj.npy")
        if not (os.path.exists(enc_path) and os.path.exists(proj_path)):
            pending.append(apath)

    print(f"Audio embeddings: {len(audio_files)} total, {len(audio_files) - len(pending)} cached, {len(pending)} pending.")
    if not pending:
        print("All audio embeddings cached — skipping Whisper model loading.")
        return

    # Load Whisper encoder
    print(f"Loading {MODEL_ID} (encoder only) ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    whisper_model = WhisperModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    encoder = whisper_model.encoder.to(DEVICE)
    encoder.eval()
    del whisper_model.decoder
    torch.cuda.empty_cache()
    print("Whisper encoder loaded.\n")

    # Initialise projector (random weights – to be trained later)
    projector = AudioProjector().half().to(DEVICE)
    print(f"Projector parameters: {sum(p.numel() for p in projector.parameters()):,}\n")

    # Process pending audio files
    print(f"Encoding {len(pending)} audio files...\n")

    for apath in pending:
        stem = os.path.splitext(os.path.basename(apath))[0]
        enc_path = os.path.join(AUDIO_EMB_DIR, f"{stem}_whisper.npy")
        proj_path = os.path.join(AUDIO_EMB_DIR, f"{stem}_proj.npy")

        waveform, sr = librosa.load(apath, sr=TARGET_SR)
        inputs = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
        mel = inputs.input_features.half().to(DEVICE)

        with torch.no_grad():
            encoder_out = encoder(mel).last_hidden_state
            projected = projector(encoder_out)

        enc_np = encoder_out.squeeze(0).cpu().float().numpy()
        proj_np = projected.squeeze(0).cpu().float().numpy()

        np.save(enc_path, enc_np)
        np.save(proj_path, proj_np)
        print(f"  Saved: {stem}  encoder={enc_np.shape}  projected={proj_np.shape}")

        del mel, encoder_out, projected
        torch.cuda.empty_cache()

    # Save projector weights (untrained – placeholder for fine-tuning)
    proj_ckpt = os.path.join(AUDIO_EMB_DIR, "audio_projector_init.pt")
    torch.save(projector.state_dict(), proj_ckpt)
    print(f"\nProjector weights saved to: {proj_ckpt}")
    print(f"Embeddings saved to:        {AUDIO_EMB_DIR}")

    # Free GPU memory
    del encoder, projector
    torch.cuda.empty_cache()


if __name__ == "__main__":
    encode_audio_whisper()
