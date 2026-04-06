"""LookingFace-compatible DualTalk-style benchmark port.

This adapts the original DualTalk architecture to the repository's shared
benchmark contract:

- wav2vec audio features as the temporal audio input
- raw left video frames as speaker-conditioning input
- FLAME content targets as the prediction output
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn

from benchmark.motion_transvae import BaselineSpeakerEncoder, evaluate_motion_metrics
from benchmark.targets import FLAME_CONTENT_DIM
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


def _masked_fill_sequence(tensor: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return tensor
    return tensor.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class DualSpeakerJointEncoder(nn.Module):
    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        feature_dim: int = 256,
        video_chunk_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.speaker_encoder = BaselineSpeakerEncoder(audio_dim=audio_dim, feature_dim=feature_dim)
        self.primary_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.partner_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.motion_context_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        del video_chunk_size

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_feature = self.speaker_encoder(
            left_video_frames=left_video_frames,
            left_audio_feat=left_audio_feat,
            padding_mask=padding_mask,
        )
        primary_audio = self.primary_proj(shared_feature)
        partner_audio = self.partner_proj(shared_feature)
        motion_context = self.motion_context_proj(shared_feature)
        primary_audio = _masked_fill_sequence(primary_audio, padding_mask)
        partner_audio = _masked_fill_sequence(partner_audio, padding_mask)
        motion_context = _masked_fill_sequence(motion_context, padding_mask)
        return primary_audio, partner_audio, motion_context


class CrossModalTemporalEnhancer(nn.Module):
    def __init__(self, feature_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        partner_audio: torch.Tensor,
        motion_context: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cross_modal_feature, _ = self.cross_attention(
            query=partner_audio,
            key=motion_context,
            value=motion_context,
            key_padding_mask=padding_mask,
        )
        temporal_feature, _ = self.temporal_lstm(cross_modal_feature)
        temporal_feature = self.output_norm(temporal_feature)
        return _masked_fill_sequence(temporal_feature, padding_mask)


class DualSpeakerInteractionModule(nn.Module):
    def __init__(self, feature_dim: int = 256, n_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        interaction_dim = feature_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=interaction_dim,
            nhead=n_heads,
            dim_feedforward=interaction_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=interaction_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        primary_audio: torch.Tensor,
        temporal_feature: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        combined_feature = torch.cat([primary_audio, temporal_feature], dim=-1)
        encoded_feature = self.interaction_encoder(combined_feature, src_key_padding_mask=padding_mask)
        enhanced_feature, _ = self.self_attention(
            encoded_feature,
            encoded_feature,
            encoded_feature,
            key_padding_mask=padding_mask,
        )
        enhanced_feature = self.dropout(encoded_feature + enhanced_feature)
        return _masked_fill_sequence(enhanced_feature, padding_mask)


class ExpressiveSynthesisModule(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        output_dim: int = FLAME_CONTENT_DIM,
        n_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        modulation_factor: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.synthesis_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.modulation_factor = modulation_factor
        self.modulation_layer = nn.Linear(feature_dim, feature_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, interaction_feature: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        decoded_feature = self.synthesis_decoder(
            tgt=interaction_feature,
            memory=interaction_feature,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        modulation = self.modulation_layer(decoded_feature)
        modulated_feature = decoded_feature + self.modulation_factor * modulation
        output = self.output_head(modulated_feature)
        return _masked_fill_sequence(output, padding_mask)


class LookingFaceDualTalk(nn.Module):
    """DualTalk-inspired content predictor adapted to raw-video plus wav2vec conditioning."""

    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        output_dim: int = FLAME_CONTENT_DIM,
        feature_dim: int = 256,
        n_heads: int = 8,
        interaction_layers: int = 3,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        video_chunk_size: int = 8,
        modulation_factor: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.joint_encoder = DualSpeakerJointEncoder(
            audio_dim=audio_dim,
            feature_dim=feature_dim,
            video_chunk_size=video_chunk_size,
            dropout=dropout,
        )
        self.temporal_enhancer = CrossModalTemporalEnhancer(
            feature_dim=feature_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.interaction_module = DualSpeakerInteractionModule(
            feature_dim=feature_dim,
            n_heads=n_heads,
            num_layers=interaction_layers,
            dropout=dropout,
        )
        self.synthesis_module = ExpressiveSynthesisModule(
            feature_dim=feature_dim * 2,
            output_dim=output_dim,
            n_heads=n_heads,
            num_layers=decoder_layers,
            dropout=dropout,
            modulation_factor=modulation_factor,
        )

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        primary_audio, partner_audio, motion_context = self.joint_encoder(
            left_audio_feat=left_audio_feat,
            left_video_frames=left_video_frames,
            padding_mask=padding_mask,
        )
        temporal_feature = self.temporal_enhancer(
            partner_audio=partner_audio,
            motion_context=motion_context,
            padding_mask=padding_mask,
        )
        interaction_feature = self.interaction_module(
            primary_audio=primary_audio,
            temporal_feature=temporal_feature,
            padding_mask=padding_mask,
        )
        prediction = self.synthesis_module(interaction_feature, padding_mask=padding_mask)
        del lengths
        return prediction, None

    def get_model_name(self) -> str:
        return "LookingFaceDualTalk"


@dataclass
class DualTalkLoss:
    w_exp: float = 1.0
    w_jaw: float = 1.0
    w_neck: float = 1.0
    w_eyes: float = 1.0
    vel_weight: float = 0.5

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        sq_error = (prediction - target) ** 2

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            denom = valid_mask.sum().clamp_min(1.0) * value.shape[-1]
            return (value * valid_mask).sum() / denom

        exp_loss = masked_mean(sq_error[:, :, :100])
        jaw_loss = masked_mean(sq_error[:, :, 100:103])
        neck_loss = masked_mean(sq_error[:, :, 103:106])
        eyes_loss = masked_mean(sq_error[:, :, 106:112])
        rec_loss = self.w_exp * exp_loss + self.w_jaw * jaw_loss + self.w_neck * neck_loss + self.w_eyes * eyes_loss

        if prediction.shape[1] > 1:
            velocity_mask = (~padding_mask[:, 1:] & ~padding_mask[:, :-1]).unsqueeze(-1).float()
            pred_velocity = prediction[:, 1:] - prediction[:, :-1]
            target_velocity = target[:, 1:] - target[:, :-1]
            vel_loss = ((pred_velocity - target_velocity) ** 2 * velocity_mask).sum()
            vel_loss = vel_loss / (velocity_mask.sum().clamp_min(1.0) * prediction.shape[-1])
        else:
            vel_loss = prediction.new_tensor(0.0)

        total = rec_loss + self.vel_weight * vel_loss
        return total, {
            "loss_total": float(total.item()),
            "loss_rec": float(rec_loss.item()),
            "loss_vel": float(vel_loss.item()),
            "loss_exp": float(exp_loss.item()),
            "loss_jaw": float(jaw_loss.item()),
            "loss_neck": float(neck_loss.item()),
            "loss_eyes": float(eyes_loss.item()),
        }


def train_dualtalk(
    model: LookingFaceDualTalk,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: DualTalkLoss,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    num_epochs: int | None = None,
    log_interval: int = 1,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = batch["flame_target_content"].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
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
def validate_dualtalk(
    model: LookingFaceDualTalk,
    loader,
    criterion: DualTalkLoss,
    device: torch.device,
    use_amp: bool = False,
    epoch: int | None = None,
    num_epochs: int | None = None,
    eval_label: str = "val",
    log_interval: int = 1,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = batch["flame_target_content"].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
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
def evaluate_dualtalk_metrics(
    model: LookingFaceDualTalk,
    loader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    class _DualTalkWrapper(nn.Module):
        def __init__(self, inner: LookingFaceDualTalk):
            super().__init__()
            self.inner = inner

        def forward(self, left_audio_feat, left_video_feat, lengths, padding_mask=None):
            prediction, _ = self.inner(
                left_audio_feat=left_audio_feat,
                left_video_frames=left_video_feat,
                lengths=lengths,
                padding_mask=padding_mask,
            )
            return prediction, None

    return evaluate_motion_metrics(
        _DualTalkWrapper(model),
        loader,
        device=device,
        target_variant="content",
        use_amp=use_amp,
    )