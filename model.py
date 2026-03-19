"""
Audio-Visual Fusion model for FLAME parameter prediction.

Architecture adapted from HERO (https://github.com/JackYu6/HERO_release):
- CrossattBlock for audio-video cross-attention fusion
- Sinusoidal PositionalEncoding
- TransformerEncoder for temporal modeling
- Separate regression heads per FLAME parameter group
"""

import math

import torch
import torch.nn as nn

from config import (
    WHISPER_DIM, QWEN_DIM, LATENT_DIM, N_HEADS,
    N_ENCODER_LAYERS, N_FUSION_BLOCKS, N_DECODER_LAYERS,
    FF_DIM, DROPOUT,
    FLAME_EXPR_DIM, FLAME_JAW_DIM, FLAME_ROT_DIM,
    FLAME_NECK_DIM, FLAME_EYES_DIM, FLAME_TRANSLATION_DIM,
)


# ── Sinusoidal Positional Encoding (from HERO) ───────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in HERO's transformer blocks."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Cross-Attention Block (adapted from HERO CrossattBlock) ──────────────────

class CrossAttention(nn.Module):
    """Multi-head cross-attention: query attends to key/value memory."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query:  (B, T_q, D)  — audio features
        memory: (B, T_m, D)  — video features (T_m=1 for single embedding)
        """
        out, _ = self.attn(
            query, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        return out


class CrossattBlock(nn.Module):
    """
    Cross-attention fusion block adapted from HERO.

    Structure: LayerNorm → CrossAttention → Residual → LayerNorm → MLP → Residual
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention with residual
        x = x + self.cross_attn(self.norm1(x), memory, memory_key_padding_mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ── FLAME Prediction Heads ───────────────────────────────────────────────────

class FLAMEHead(nn.Module):
    """Separate linear heads for each FLAME parameter group."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.expr_head = nn.Linear(d_model, FLAME_EXPR_DIM)
        self.jaw_head = nn.Linear(d_model, FLAME_JAW_DIM)
        self.rot_head = nn.Linear(d_model, FLAME_ROT_DIM)
        self.neck_head = nn.Linear(d_model, FLAME_NECK_DIM)
        self.eyes_head = nn.Linear(d_model, FLAME_EYES_DIM)
        self.trans_head = nn.Linear(d_model, FLAME_TRANSLATION_DIM)
        self._init_translation_head()

    def _init_translation_head(self) -> None:
        """Small init keeps early translation predictions stable near zero."""
        nn.init.normal_(self.trans_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.trans_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, T, D)
        Returns dict of (B, T, param_dim) tensors.
        """
        h = self.pre(x)
        return {
            "expr": self.expr_head(h),
            "jaw_pose": self.jaw_head(h),
            "rotation": self.rot_head(h),
            "neck_pose": self.neck_head(h),
            "eyes_pose": self.eyes_head(h),
            "translation": self.trans_head(h),
        }


# ── Main Model ───────────────────────────────────────────────────────────────

class AudioVisualFLAMEModel(nn.Module):
    """
    Audio-visual fusion model for per-frame FLAME parameter prediction.

    Audio (temporal) features are processed with self-attention, then fused
    with video (global identity) features via cross-attention blocks adapted
    from HERO. A temporal decoder further refines the fused representation
    before separate heads predict each FLAME parameter group.
    """

    def __init__(
        self,
        audio_dim: int = WHISPER_DIM,
        video_dim: int = QWEN_DIM,
        latent_dim: int = LATENT_DIM,
        n_heads: int = N_HEADS,
        n_encoder_layers: int = N_ENCODER_LAYERS,
        n_fusion_blocks: int = N_FUSION_BLOCKS,
        n_decoder_layers: int = N_DECODER_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # ── Audio encoder ─────────────────────────────────────────────────
        self.audio_proj = nn.Linear(audio_dim, latent_dim)
        self.audio_pos_enc = PositionalEncoding(latent_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        # ── Video encoder ─────────────────────────────────────────────────
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        # ── Fusion blocks (HERO-style cross-attention) ────────────────────
        self.fusion_blocks = nn.ModuleList([
            CrossattBlock(latent_dim, n_heads, ff_dim, dropout)
            for _ in range(n_fusion_blocks)
        ])

        # ── Temporal decoder ──────────────────────────────────────────────
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=n_decoder_layers
        )

        # ── FLAME prediction heads ────────────────────────────────────────
        self.flame_head = FLAMEHead(latent_dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio_feat: (B, T, 1280) — temporally-aligned Whisper features
            video_feat: (B, 3584)    — Qwen video embedding
            src_key_padding_mask: (B, T) — True for padded positions

        Returns:
            Dict of FLAME parameter predictions, each (B, T, param_dim).
        """
        # Encode audio: project + positional encoding + self-attention
        audio = self.audio_proj(audio_feat)              # (B, T, D)
        audio = self.audio_pos_enc(audio)                # (B, T, D)
        audio = self.audio_encoder(
            audio, src_key_padding_mask=src_key_padding_mask
        )                                                # (B, T, D)

        # Encode video: project to single token
        video = self.video_proj(video_feat)              # (B, D)
        video = video.unsqueeze(1)                       # (B, 1, D)

        # Fuse via cross-attention (audio queries, video memory)
        fused = audio
        for block in self.fusion_blocks:
            fused = block(fused, video)                  # (B, T, D)

        # Temporal decoder for further refinement
        fused = self.temporal_decoder(
            fused, src_key_padding_mask=src_key_padding_mask
        )                                                # (B, T, D)

        # Predict FLAME parameters
        return self.flame_head(fused)
