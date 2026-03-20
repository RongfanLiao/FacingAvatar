"""Configuration for Audio-Visual FLAME parameter prediction."""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
AUDIO_EMB_DIR = os.path.join(DATA_DIR, "audio_embeddings")
VIDEO_EMB_DIR = os.path.join(DATA_DIR, "video_embeddings")
VIDEO_LABELS_DIR = os.path.join(DATA_DIR, "video_labels")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── Input dimensions ──────────────────────────────────────────────────────────
WHISPER_DIM = 1280       # Whisper-Large-v3 encoder hidden size
QWEN_DIM = 3584          # Qwen2.5-VL-7B hidden size
WHISPER_MAX_FRAMES = 1500  # Whisper always outputs 1500 frames (30s chunks)
WHISPER_CHUNK_SEC = 30.0   # Whisper processes 30-second chunks
AUDIO_SR = 16000           # Audio sample rate

# ── FLAME output dimensions (identity-agnostic, no shape) ─────────────────────
FLAME_EXPR_DIM = 100
FLAME_JAW_DIM = 3
FLAME_ROT_DIM = 3
FLAME_NECK_DIM = 3
FLAME_EYES_DIM = 6
FLAME_TRANSLATION_DIM = 3
FLAME_SHAPE_DIM = 300
FLAME_TOTAL_DIM = (FLAME_EXPR_DIM + FLAME_JAW_DIM + FLAME_ROT_DIM
                   + FLAME_NECK_DIM + FLAME_EYES_DIM + FLAME_TRANSLATION_DIM)  # 118

# ── Model architecture ────────────────────────────────────────────────────────
LATENT_DIM = 256
N_HEADS = 8
N_ENCODER_LAYERS = 4     # Audio self-attention layers
N_FUSION_BLOCKS = 2      # Cross-attention fusion blocks
N_DECODER_LAYERS = 4     # Post-fusion temporal decoder layers
FF_DIM = 512             # Feed-forward hidden dim
DROPOUT = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
LR = 2e-4
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.99)
BATCH_SIZE = 4
NUM_EPOCHS = 500
WARMUP_STEPS = 500
VEL_LOSS_WEIGHT = 0.5    # Velocity regularization weight

# Per-parameter loss factors: higher expr weight improves expression sensitivity,
# lower translation weight keeps it learnable but prevents jitter from dominating.
PARAM_LOSS_WEIGHTS = {
    "expr": 2.0,
    "jaw_pose": 1.0,
    "rotation": 1.0,
    "neck_pose": 1.0,
    "eyes_pose": 1.0,
    "translation": 0.1,
}
VEL_PARAM_LOSS_WEIGHTS = {
    "expr": 2.0,
    "jaw_pose": 1.0,
    "rotation": 1.0,
    "neck_pose": 1.0,
    "eyes_pose": 1.0,
    "translation": 0.1,
}

NUM_WORKERS = 4
VAL_SEQS = ["2928", "2933", "2935_s"]  # Hold-out validation sequences
TRAIN_VAL_SAME_SEQS = True  # Convergence test mode: train and val use all sequences
# CONVERGENCE_SEQ_IDS = ["2920", "2921", "2922", "2923", "2924"]  # Empty list -> use all discovered
CONVERGENCE_SEQ_IDS = []  # Empty list -> use all discovered

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"
