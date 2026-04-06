"""Dyadic-Interaction-Modeling ContinuousTransformer port for LookingFace.

This adapts the ContinuousTransformer family from ../Dyadic-Interaction-Modeling
to the shared benchmark contract used by motion_transvae in this repository:

- wav2vec audio features
- raw left video frames
- FLAME content or motion58 targets
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn

from benchmark.motion_transvae import BaselineSpeakerEncoder, PositionalEncoding, evaluate_motion_metrics
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM
from config import WAV2VEC_DIM


def _format_progress_logs(logs: dict[str, float], prefix: str) -> str:
    ordered_keys = [
        "loss_total",
        "loss_rec",
        "loss_vel",
        "loss_exp",
        "loss_jaw",
        "loss_neck",
        "loss_eyes",
        "loss_rot",
        "loss_tran",
    ]
    parts = [prefix]
    for key in ordered_keys:
        if key in logs:
            parts.append(f"{key}={logs[key]:.5f}")
    return " | ".join(parts)


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress_prefix(
    label: str,
    batch_idx: int,
    total_batches: int,
    start_time: float,
    epoch: int | None = None,
    num_epochs: int | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_seconds = time.perf_counter() - start_time
    elapsed = _format_elapsed(elapsed_seconds)
    progress = (batch_idx / total_batches * 100.0) if total_batches > 0 else 0.0
    avg_seconds = elapsed_seconds / batch_idx if batch_idx > 0 else 0.0
    remaining_batches = max(total_batches - batch_idx, 0)
    eta = _format_elapsed(avg_seconds * remaining_batches)
    if epoch is not None and num_epochs is not None:
        return (
            f"[{timestamp}] [{label}] epoch {epoch}/{num_epochs} "
            f"iter {batch_idx}/{total_batches} ({progress:.1f}%) elapsed={elapsed} eta={eta}"
        )
    return f"[{timestamp}] [{label}] iter {batch_idx}/{total_batches} ({progress:.1f}%) elapsed={elapsed} eta={eta}"


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class DyadicContinuousTransformer(nn.Module):
    """ContinuousTransformer-style seq2seq model adapted to the local contract."""

    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        target_variant: str = "content",
        feature_dim: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        video_chunk_size: int = 8,
    ):
        super().__init__()
        if target_variant == "content":
            self.output_dim = FLAME_CONTENT_DIM
        elif target_variant == "motion58":
            self.output_dim = FLAME_58_DIM
        else:
            raise ValueError(f"Unsupported target_variant: {target_variant}")

        self.target_variant = target_variant
        self.context_encoder = BaselineSpeakerEncoder(audio_dim=audio_dim, feature_dim=feature_dim)
        self.src_pos = PositionalEncoding(feature_dim, dropout=dropout)
        self.tgt_pos = PositionalEncoding(feature_dim, dropout=dropout)
        self.target_proj = nn.Linear(self.output_dim, feature_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, self.output_dim),
        )
        self.video_chunk_size = video_chunk_size

    def encode_source(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        context = self.context_encoder(left_video_frames, left_audio_feat, padding_mask=padding_mask)
        context = self.src_pos(context)
        return self.encoder(context, src_key_padding_mask=padding_mask)

    def decode_teacher_forced(
        self,
        memory: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shifted_target = torch.cat([self.start_token.expand(target.shape[0], -1, -1), target[:, :-1]], dim=1)
        decoder_tokens = self.target_proj(shifted_target)
        decoder_tokens = self.tgt_pos(decoder_tokens)
        tgt_mask = _causal_mask(decoder_tokens.shape[1], decoder_tokens.device)
        decoded = self.decoder(
            tgt=decoder_tokens,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        return self.output_head(decoded)

    def decode_autoregressive(
        self,
        memory: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = memory.shape
        generated = self.start_token.expand(batch_size, 1, -1)
        outputs: list[torch.Tensor] = []
        for _ in range(seq_len):
            decoder_tokens = self.target_proj(generated)
            decoder_tokens = self.tgt_pos(decoder_tokens)
            tgt_mask = _causal_mask(decoder_tokens.shape[1], decoder_tokens.device)
            decoded = self.decoder(
                tgt=decoder_tokens,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=padding_mask,
            )
            next_frame = self.output_head(decoded[:, -1:])
            outputs.append(next_frame)
            generated = torch.cat([generated, next_frame], dim=1)

        prediction = torch.cat(outputs, dim=1)
        if padding_mask is not None:
            prediction = prediction.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        del lengths
        return prediction

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        memory = self.encode_source(left_audio_feat, left_video_frames, padding_mask)
        if target is not None:
            prediction = self.decode_teacher_forced(memory, target, padding_mask)
        else:
            prediction = self.decode_autoregressive(memory, lengths, padding_mask)

        if padding_mask is not None:
            prediction = prediction.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return prediction, None

    def get_model_name(self) -> str:
        return "DyadicContinuousTransformer"


@dataclass
class DyadicContinuousTransformerLoss:
    """Masked regression loss for target variants matching motion_transvae."""

    target_variant: str = "content"
    w_exp: float = 2.0
    w_jaw: float = 2.0
    w_neck: float = 2.0
    w_eyes: float = 2.0
    w_rot: float = 4.0
    w_tran: float = 4.0
    vel_weight: float = 0.5

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        mse = (prediction - target) ** 2

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            denom = valid_mask.sum().clamp_min(1.0) * value.shape[-1]
            return (value * valid_mask).sum() / denom

        if self.target_variant == "motion58":
            exp_loss = masked_mean(mse[:, :, :52])
            rot_loss = masked_mean(mse[:, :, 52:55])
            tran_loss = masked_mean(mse[:, :, 55:58])
            rec_loss = self.w_exp * exp_loss + self.w_rot * rot_loss + self.w_tran * tran_loss
            component_logs = {
                "loss_exp": float(exp_loss.item()),
                "loss_rot": float(rot_loss.item()),
                "loss_tran": float(tran_loss.item()),
            }
        elif self.target_variant == "content":
            exp_loss = masked_mean(mse[:, :, :100])
            jaw_loss = masked_mean(mse[:, :, 100:103])
            neck_loss = masked_mean(mse[:, :, 103:106])
            eyes_loss = masked_mean(mse[:, :, 106:112])
            rec_loss = self.w_exp * exp_loss + self.w_jaw * jaw_loss + self.w_neck * neck_loss + self.w_eyes * eyes_loss
            component_logs = {
                "loss_exp": float(exp_loss.item()),
                "loss_jaw": float(jaw_loss.item()),
                "loss_neck": float(neck_loss.item()),
                "loss_eyes": float(eyes_loss.item()),
            }
        else:
            raise ValueError(f"Unsupported target_variant: {self.target_variant}")

        if prediction.shape[1] > 1:
            valid_velocity = (~padding_mask[:, 1:] & ~padding_mask[:, :-1]).unsqueeze(-1).float()
            pred_vel = prediction[:, 1:] - prediction[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            vel_loss = ((pred_vel - target_vel) ** 2 * valid_velocity).sum() / (valid_velocity.sum().clamp_min(1.0) * prediction.shape[-1])
        else:
            vel_loss = prediction.new_tensor(0.0)

        total = rec_loss + self.vel_weight * vel_loss
        return total, {
            "loss_total": float(total.item()),
            "loss_rec": float(rec_loss.item()),
            "loss_vel": float(vel_loss.item()),
            **component_logs,
        }


def _resolve_target(batch: dict[str, torch.Tensor], target_variant: str) -> torch.Tensor:
    return batch["flame_target_58"] if target_variant == "motion58" else batch["flame_target_content"]


def train_dyadic_dim(
    model: DyadicContinuousTransformer,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: DyadicContinuousTransformerLoss,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    num_epochs: int | None = None,
    log_interval: int = 1,
) -> dict[str, float]:
    """One training epoch for the Dyadic ContinuousTransformer port."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = _resolve_target(batch, criterion.target_variant).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
                target=target,
            )
            loss, logs = criterion(prediction, target, padding_mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            prefix = _format_progress_prefix(
                label="train",
                batch_idx=batch_idx,
                total_batches=total_batches,
                start_time=epoch_start_time,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            print(_format_progress_logs(logs, prefix), flush=True)

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_dyadic_dim(
    model: DyadicContinuousTransformer,
    loader,
    criterion: DyadicContinuousTransformerLoss,
    device: torch.device,
    use_amp: bool = False,
    epoch: int | None = None,
    num_epochs: int | None = None,
    eval_label: str = "val",
    log_interval: int = 1,
) -> dict[str, float]:
    """One validation epoch for the Dyadic ContinuousTransformer port."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = _resolve_target(batch, criterion.target_variant).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
                target=target,
            )
            _, logs = criterion(prediction, target, padding_mask)

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            prefix = _format_progress_prefix(
                label=eval_label,
                batch_idx=batch_idx,
                total_batches=total_batches,
                start_time=eval_start_time,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            print(_format_progress_logs(logs, prefix), flush=True)

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_dyadic_dim_metrics(
    model: DyadicContinuousTransformer,
    loader,
    device: torch.device,
    target_variant: str = "content",
    use_amp: bool = False,
) -> dict[str, float]:
    """Evaluate the port with the shared motion metric stack."""

    class _DyadicWrapper(nn.Module):
        def __init__(self, inner: DyadicContinuousTransformer):
            super().__init__()
            self.inner = inner

        def forward(self, left_audio_feat, left_video_feat, lengths, padding_mask=None):
            prediction, _ = self.inner(
                left_audio_feat=left_audio_feat,
                left_video_frames=left_video_feat,
                lengths=lengths,
                padding_mask=padding_mask,
                target=None,
            )
            return prediction, None

    return evaluate_motion_metrics(
        _DyadicWrapper(model),
        loader,
        device=device,
        target_variant=target_variant,
        use_amp=use_amp,
    )