"""LookingFace-compatible motion-only port of the baseline motion_transvae model."""

from __future__ import annotations

import math
import os
import time
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from benchmark.targets import FLAME_118_DIM, FLAME_CONTENT_DIM, flame_component_layout, flame_npz_to_targets, flame_target_key, flame_target_variant
from config import REACTION_ANNOTATIONS_DIR, VIDEO_CANVAS_SIZE, WAV2VEC_DIM


TYPE_METRIC_CACHE_VERSION = "cheap-dtw-frdist-v1"


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Build a boolean mask where True denotes valid sequence positions."""
    max_len = int(max_len if max_len is not None else lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with batch-first inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def init_biased_mask(n_head: int, max_seq_len: int, period: int) -> torch.Tensor:
    """Temporal bias mask adapted from the baseline TransVAE implementation."""

    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(power: int) -> list[float]:
            start = 2 ** (-2 ** -(math.log2(power) - 3))
            ratio = start
            return [start * ratio ** i for i in range(power)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)

        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).reshape(-1) // period
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for index in range(max_seq_len):
        alibi[index, : index + 1] = bias[-(index + 1) :]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask.unsqueeze(0) + alibi


class LatentVAE(nn.Module):
    """Sequence-level variational encoder used by the decoder."""

    def __init__(self, in_channels: int, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(in_channels, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, encoded_feature: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Normal]:
        batch_size, max_len, _ = encoded_feature.shape
        valid_mask = lengths_to_mask(lengths, max_len=max_len)
        x = self.input_proj(encoded_feature)

        mu_token = self.mu_token.expand(batch_size, -1, -1)
        logvar_token = self.logvar_token.expand(batch_size, -1, -1)
        x = torch.cat([mu_token, logvar_token, x], dim=1)

        token_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=encoded_feature.device)
        src_key_padding_mask = ~torch.cat([token_mask, valid_mask], dim=1)
        x = self.sequence_encoder(x, src_key_padding_mask=src_key_padding_mask)

        mu = x[:, 0]
        logvar = x[:, 1]
        std = torch.exp(0.5 * logvar)
        distribution = torch.distributions.Normal(mu, std)
        motion_sample = distribution.rsample()
        return motion_sample, distribution


class MotionDecoder(nn.Module):
    """Transformer decoder that predicts FLAME content targets."""

    def __init__(self, output_dim: int = FLAME_118_DIM, feature_dim: int = 128, n_head: int = 4, max_seq_len: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.n_head = n_head
        self.vae = LatentVAE(feature_dim, feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=n_head,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
        )
        self.decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.max_seq_len = max_seq_len
        self.register_buffer("biased_mask", init_biased_mask(n_head=n_head, max_seq_len=max_seq_len, period=max_seq_len))
        self.position = PositionalEncoding(feature_dim)
        self.output_head = nn.Linear(feature_dim, output_dim)

    def _get_tgt_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self.biased_mask = init_biased_mask(
                n_head=self.n_head,
                max_seq_len=self.max_seq_len,
                period=self.max_seq_len,
            ).to(device)
        mask = self.biased_mask[:, :seq_len, :seq_len].to(device)
        return mask.repeat(batch_size, 1, 1)

    def forward(self, encoded_feature: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Normal]:
        batch_size, seq_len, _ = encoded_feature.shape
        motion_sample, distribution = self.vae(encoded_feature, lengths)
        time_queries = torch.zeros(batch_size, seq_len, self.feature_dim, device=encoded_feature.device)
        time_queries = self.position(time_queries)
        tgt_mask = self._get_tgt_mask(seq_len, batch_size, encoded_feature.device)
        decoded = self.decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        decoded = self.decoder_2(decoded, decoded, tgt_mask=tgt_mask)
        return self.output_head(decoded), distribution


class ConvBlock(nn.Module):
    """Conv3D block for temporal video feature extraction.

    Ported from baseline_react2025/framework/motion_transvae/BasicBlock.py.
    Input:  (B, 3, T, H, W)
    Output: (B, planes, T)  via spatial global average pooling.
    """

    def __init__(self, in_planes: int = 3, planes: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes // 4, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.in1 = nn.InstanceNorm3d(planes // 4)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.conv2 = nn.Conv3d(planes // 4, planes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.in2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.in3 = nn.InstanceNorm3d(planes)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.in4 = nn.InstanceNorm3d(planes)
        self.conv5 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.in5 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.in1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = x + self.relu(self.in4(self.conv4(x)))
        x = self.relu(self.in5(self.conv5(x)))
        return x.mean(dim=-1).mean(dim=-1)  # (B, planes, T)


class VideoEncoder(nn.Module):
    """Conv3D video encoder that maps raw frames to per-frame feature vectors.

    Ported from baseline_react2025/framework/motion_transvae/TransformerVAE.py.
    Input:  (B, T, 3, H, W)
    Output: (B, T, feature_dim)

    Since all Conv3D kernels have temporal size 1, frames are processed
    independently.  To avoid OOM on long sequences, the forward pass chunks
    the temporal dimension into groups of ``chunk_size`` frames and
    concatenates the results.
    """

    def __init__(self, feature_dim: int = 128, chunk_size: int = 300):
        super().__init__()
        self.conv3d = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.chunk_size = chunk_size

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        if T <= self.chunk_size:
            video_input = video.transpose(1, 2)  # (B, 3, T, H, W)
            token_output = self.conv3d(video_input).transpose(1, 2)
            return self.fc(token_output)

        # Process in temporal chunks to limit peak GPU memory
        outputs = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            chunk = video[:, t0:t1].transpose(1, 2)  # (B, 3, chunk_len, H, W)
            out = self.conv3d(chunk).transpose(1, 2)  # (B, chunk_len, feature_dim)
            outputs.append(out)
        token_output = torch.cat(outputs, dim=1)  # (B, T, feature_dim)
        return self.fc(token_output)


class BaselineSpeakerEncoder(nn.Module):
    """Speaker encoder that fuses raw video frames + wav2vec audio via concat+linear.

    Follows the SpeakerBehaviourEncoder pattern from baseline_react2025.
    """

    def __init__(self, audio_dim: int = WAV2VEC_DIM, feature_dim: int = 128):
        super().__init__()
        self.video_encoder = VideoEncoder(feature_dim=feature_dim)
        self.audio_feature_map = nn.Linear(audio_dim, feature_dim)
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

    def forward(
        self,
        left_video_frames: torch.Tensor,
        left_audio_feat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        video_feature = self.video_encoder(left_video_frames)  # (B, T, feature_dim)
        audio_feature = self.audio_feature_map(left_audio_feat)  # (B, T, feature_dim)
        return self.fusion_layer(torch.cat([video_feature, audio_feature], dim=-1))


class SpeakerContextEncoder(nn.Module):
    """LookingFace-specific encoder for left audio and static left video embeddings."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        feature_dim: int,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, feature_dim)
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.position = PositionalEncoding(feature_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio_tokens = self.audio_proj(left_audio_feat)
        audio_tokens = self.position(audio_tokens)
        audio_tokens = self.audio_encoder(audio_tokens, src_key_padding_mask=padding_mask)

        video_token = self.video_proj(left_video_feat).unsqueeze(1)
        attended_video, _ = self.cross_attention(audio_tokens, video_token, video_token)
        fused = self.fusion(torch.cat([audio_tokens, attended_video], dim=-1))
        return fused


class MotionTransformerVAE(nn.Module):
    """Motion-only port of baseline motion_transvae for LookingFace benchmarking.

    Uses BaselineSpeakerEncoder (Conv3D video + wav2vec audio, concat+linear fusion)
    following the baseline_react2025 architecture.
    """

    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        output_dim: int = FLAME_118_DIM,
        feature_dim: int = 128,
        n_heads: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.context_encoder = BaselineSpeakerEncoder(
            audio_dim=audio_dim,
            feature_dim=feature_dim,
        )
        self.decoder = MotionDecoder(
            output_dim=output_dim,
            feature_dim=feature_dim,
            n_head=n_heads,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.distributions.Normal]:
        encoded_feature = self.context_encoder(left_video_frames, left_audio_feat, padding_mask=padding_mask)
        return self.decoder(encoded_feature, lengths)

    def get_model_name(self) -> str:
        return "MotionOnlyTransformerVAE"


@dataclass
class MotionVAELoss:
    """Masked reconstruction plus KL loss for FLAME content targets."""

    kl_p: float = 1e-5
    w_exp: float = 2.0
    w_jaw: float = 2.0
    w_rot: float = 1.0
    w_neck: float = 2.0
    w_eyes: float = 2.0
    w_tran: float = 0.1

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        distribution: torch.distributions.Normal,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        mse = (prediction - target) ** 2

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            denom = valid_mask.sum().clamp_min(1.0) * value.shape[-1]
            return (value * valid_mask).sum() / denom

        component_logs: dict[str, float] = {}
        rec_loss = prediction.new_tensor(0.0)
        for name, component_slice in flame_component_layout(prediction.shape[-1]):
            component_loss = masked_mean(mse[:, :, component_slice])
            rec_loss = rec_loss + getattr(self, f"w_{name}") * component_loss
            component_logs[f"loss_{name}"] = float(component_loss.item())

        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kld_loss = torch.distributions.kl_divergence(distribution, distribution_ref).mean()
        loss = rec_loss + self.kl_p * kld_loss

        return loss, {
            "loss_total": float(loss.item()),
            "loss_rec": float(rec_loss.item()),
            "loss_kld": float(kld_loss.item()),
            **component_logs,
        }


def diversity_loss(prediction_a: torch.Tensor, prediction_b: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Sequence diversity penalty adapted from the baseline TransVAE training loop."""
    valid_mask = (~padding_mask).unsqueeze(-1)
    flat_a = prediction_a[valid_mask.expand_as(prediction_a)].reshape(1, -1)
    flat_b = prediction_b[valid_mask.expand_as(prediction_b)].reshape(1, -1)
    stacked = torch.cat([flat_a, flat_b], dim=0).float()
    dist2 = torch.pdist(stacked, p=2) ** 2
    return torch.exp(-dist2 / 100.0).mean()


def _format_progress_logs(logs: dict[str, float], prefix: str) -> str:
    ordered_keys = ["loss_total_with_div", "loss_total", "loss_rec", "loss_kld", "loss_div"]
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


def train_motion_transvae(
    model,
    loader,
    optimizer,
    criterion,
    device,
    div_p: float = 10.0,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    num_epochs: int | None = None,
    log_interval: int = 1,
) -> dict[str, float]:
    """One training epoch for the LookingFace motion-only TransVAE port."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    use_amp = scaler is not None
    total_batches = len(loader)
    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target = batch[flame_target_key(model.decoder.output_dim)].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, distribution = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)
            loss, logs = criterion(prediction, target, distribution, padding_mask)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction_b, _ = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)
        div = diversity_loss(prediction, prediction_b, padding_mask)
        total_loss = loss + div_p * div

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        logs["loss_div"] = float(div.item())
        logs["loss_total_with_div"] = float(total_loss.item())
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
def validate_motion_transvae(
    model,
    loader,
    criterion,
    device,
    use_amp: bool = False,
    epoch: int | None = None,
    num_epochs: int | None = None,
    eval_label: str = "val",
    log_interval: int = 1,
) -> dict[str, float]:
    """Validation loop for the LookingFace motion-only TransVAE port."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target = batch[flame_target_key(model.decoder.output_dim)].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, distribution = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)
            _, logs = criterion(prediction, target, distribution, padding_mask)
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


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root for symmetric positive semi-definite matrices."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Fréchet distance between two feature sets."""
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        return float("nan")

    mu_a = features_a.mean(axis=0)
    mu_b = features_b.mean(axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    mean_term = np.sum((mu_a - mu_b) ** 2)
    sigma_a = np.atleast_2d(sigma_a)
    sigma_b = np.atleast_2d(sigma_b)
    cov_prod = _sqrtm_psd(sigma_a) @ sigma_b @ _sqrtm_psd(sigma_a)
    trace_term = np.trace(sigma_a + sigma_b - 2.0 * _sqrtm_psd(cov_prod))
    return float(mean_term + trace_term)


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    matrix = np.cov(x, y)
    try:
        diag = np.diag(matrix)
    except ValueError:
        return float(matrix / matrix)
    stddev = np.sqrt(diag.real)
    denom = np.outer(stddev, stddev)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
    corr = np.nan_to_num(corr)
    return float(np.clip(corr[0, 1], -1.0, 1.0))


def _concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Baseline-compatible CCC averaged over feature dimensions."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    ccc_list = []
    for dim in range(y_true.shape[1]):
        cor = _corrcoef(y_true[:, dim], y_pred[:, dim])
        mean_true = np.mean(y_true[:, dim])
        mean_pred = np.mean(y_pred[:, dim])
        var_true = np.var(y_true[:, dim])
        var_pred = np.var(y_pred[:, dim])
        sd_true = np.std(y_true[:, dim])
        sd_pred = np.std(y_pred[:, dim])
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / (denominator + 1e-8)
        ccc_list.append(float(ccc))
    return float(np.mean(ccc_list))


def _dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    """Simple DTW implementation for 2D motion sequences."""
    len_a, len_b = sequence_a.shape[0], sequence_b.shape[0]
    dp = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for index_a in range(1, len_a + 1):
        for index_b in range(1, len_b + 1):
            cost = np.linalg.norm(sequence_a[index_a - 1] - sequence_b[index_b - 1])
            dp[index_a, index_b] = cost + min(
                dp[index_a - 1, index_b],
                dp[index_a, index_b - 1],
                dp[index_a - 1, index_b - 1],
            )

    return float(dp[len_a, len_b])


def _motion_frd(pred_seq: np.ndarray, target_seq: np.ndarray) -> float:
    """Full FLAME FRD over the 118-d target using grouped DTW by component."""
    groups = [
        (component_slice.start, component_slice.stop, 1.0 / float(component_slice.stop - component_slice.start))
        for _, component_slice in flame_component_layout(FLAME_118_DIM)
    ]
    total = 0.0
    for start, end, weight in groups:
        total += weight * _dtw_distance(
            pred_seq[:, start:end].astype(np.float32), 
            target_seq[:, start:end].astype(np.float32)
        )
    return float(total)


def _content_frd(pred_seq: np.ndarray, target_seq: np.ndarray) -> float:
    """Content-focused FRD over expr, jaw, neck, and eyes."""
    groups = [
        (0, 100, 1.0 / 100.0),
        (100, 103, 1.0 / 3.0),
        (103, 106, 1.0 / 3.0),
        (106, 112, 1.0 / 6.0),
    ]
    total = 0.0
    for start, end, weight in groups:
        total += weight * _dtw_distance(
            pred_seq[:, start:end].astype(np.float32), 
            target_seq[:, start:end].astype(np.float32),
        )
    return float(total)


def _windowed_dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray, window_radius: int) -> float:
    """DTW with Sakoe-Chiba band constraint for cheaper approximate matching."""
    len_a, len_b = sequence_a.shape[0], sequence_b.shape[0]
    window_radius = max(int(window_radius), abs(len_a - len_b))
    dp = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for index_a in range(1, len_a + 1):
        start_b = max(1, index_a - window_radius)
        end_b = min(len_b, index_a + window_radius) + 1
        for index_b in range(start_b, end_b):
            cost = np.linalg.norm(sequence_a[index_a - 1] - sequence_b[index_b - 1])
            dp[index_a, index_b] = cost + min(
                dp[index_a - 1, index_b],
                dp[index_a, index_b - 1],
                dp[index_a - 1, index_b - 1],
            )

    return float(dp[len_a, len_b])


def _cheap_dtw_frd_against_resampled_references(
    pred_seq: np.ndarray,
    references: np.ndarray,
    feature_dim: int,
    max_frames: int = 96,
    window_ratio: float = 0.1,
) -> float:
    """Cheaper FRD approximation using downsampled, windowed DTW.

    This keeps the DTW-style accumulated path cost, but computes it on temporally
    downsampled sequences with a Sakoe-Chiba band to preserve scale better than
    simple frame-aligned averaging.
    """
    if references.size == 0:
        return float("nan")

    pred = np.asarray(pred_seq, dtype=np.float32)
    refs = np.asarray(references, dtype=np.float32)
    if pred.ndim != 2 or refs.ndim != 3:
        raise ValueError(f"Expected pred (T, D) and refs (N, T, D), got {pred.shape} and {refs.shape}")
    if refs.shape[1] != pred.shape[0] or refs.shape[2] != pred.shape[1] or pred.shape[1] != feature_dim:
        raise ValueError(
            f"Expected refs (N, {pred.shape[0]}, {feature_dim}) and pred (*, {feature_dim}), got {refs.shape} and {pred.shape}"
        )

    approx_length = min(pred.shape[0], max_frames)
    pred_down = _resample_sequence_length(pred, approx_length) if pred.shape[0] != approx_length else pred
    refs_down = np.stack([
        _resample_sequence_length(reference_seq, approx_length)
        if reference_seq.shape[0] != approx_length else reference_seq
        for reference_seq in refs
    ]).astype(np.float32, copy=False)

    window_radius = max(int(np.ceil(approx_length * window_ratio)), 2)
    path_scale = pred.shape[0] / float(approx_length)
    best_total = float("inf")

    for reference_down in refs_down:
        total = 0.0
        for _, component_slice in flame_component_layout(feature_dim):
            component_cost = _windowed_dtw_distance(
                pred_down[:, component_slice],
                reference_down[:, component_slice],
                window_radius=window_radius,
            )
            total += path_scale * component_cost / float(component_slice.stop - component_slice.start)
        if total < best_total:
            best_total = total

    return float(best_total)


def _stack_valid_sequences(items: Iterable[np.ndarray], feature_dim: int) -> np.ndarray:
    arrays = [item for item in items if item.size > 0]
    if not arrays:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def _resample_sequence_length(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a temporal sequence to a target number of frames."""
    if sequence.shape[0] == target_length:
        return sequence
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if sequence.shape[0] == 0:
        raise ValueError("Cannot resample an empty sequence")
    if sequence.shape[0] == 1:
        return np.repeat(sequence, target_length, axis=0)

    src_positions = np.linspace(0, sequence.shape[0] - 1, num=sequence.shape[0], dtype=np.float32)
    dst_positions = np.linspace(0, sequence.shape[0] - 1, num=target_length, dtype=np.float32)
    resampled = [np.interp(dst_positions, src_positions, sequence[:, dim]) for dim in range(sequence.shape[1])]
    return np.stack(resampled, axis=1).astype(np.float32)


def _normalize_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized or None


def _load_reaction_annotation(seq_id: str, annotations_dir: str) -> dict[str, object] | None:
    path = os.path.join(annotations_dir, f"{seq_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return None
    return data


def _annotation_bucket_key(seq_id: str, annotations_dir: str) -> tuple[str | None, str | None]:
    record = _load_reaction_annotation(seq_id, annotations_dir)
    if not record or record.get("status") != "ok":
        return None, None

    sample = record.get("sample")
    annotation = record.get("annotation")
    if not isinstance(sample, dict) or not isinstance(annotation, dict):
        return None, None

    content_type = _normalize_content_type(sample.get("content_type"))
    dominant = str(annotation.get("dominant_reaction_type", "")).strip().upper()
    if dominant not in set("ABCDEFGH"):
        return content_type, None
    return content_type, dominant


def _build_appropriate_reference_sets(
    reference_seq_ids: list[str],
    manifest: dict[str, dict[str, str]],
    target_key: str,
    annotations_dir: str,
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    stats = {
        "reference_candidates": 0,
        "reference_with_annotations": 0,
        "reference_with_bucket": 0,
        "reference_sequences_loaded": 0,
    }

    for seq_id in reference_seq_ids:
        stats["reference_candidates"] += 1
        content_type, dominant = _annotation_bucket_key(seq_id, annotations_dir)
        if content_type is None:
            continue
        stats["reference_with_annotations"] += 1
        if dominant is None:
            continue
        stats["reference_with_bucket"] += 1

        entry = manifest.get(seq_id)
        if entry is None:
            continue
        flame_npz = entry.get("flame_npz")
        if not flame_npz or not os.path.exists(flame_npz):
            continue

        targets = flame_npz_to_targets(flame_npz)
        grouped[(content_type, dominant)].append(np.asarray(targets[target_key], dtype=np.float32))
        stats["reference_sequences_loaded"] += 1

    return grouped, stats


def _get_resampled_references(
    bucket_key: tuple[str, str],
    references: list[np.ndarray],
    target_length: int,
    cache: dict[tuple[str, str, int], np.ndarray],
) -> np.ndarray:
    cache_key = (bucket_key[0], bucket_key[1], int(target_length))
    cached = cache.get(cache_key)
    if cached is None:
        cached = np.stack([
            _resample_sequence_length(reference_seq, target_length)
            for reference_seq in references
        ]).astype(np.float32, copy=False)
        cache[cache_key] = cached
    return cached


def _max_concordance_correlation_coefficient_against_references(
    y_pred: np.ndarray,
    references: np.ndarray,
) -> float:
    """Compute the best CCC against a bank of references with a vectorized implementation."""
    if references.size == 0:
        return float("nan")

    pred = np.asarray(y_pred, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    if refs.ndim == 2:
        refs = refs[:, :, None]

    if pred.shape[0] != refs.shape[1] or pred.shape[1] != refs.shape[2]:
        raise ValueError(
            f"Expected references with shape (N, {pred.shape[0]}, {pred.shape[1]}), got {refs.shape}"
        )

    pred_mean = pred.mean(axis=0)
    ref_mean = refs.mean(axis=1)
    pred_centered = pred - pred_mean
    ref_centered = refs - ref_mean[:, None, :]

    if pred.shape[0] > 1:
        sample_cov = (ref_centered * pred_centered[None, :, :]).sum(axis=1) / float(pred.shape[0] - 1)
        sample_var_ref = (ref_centered ** 2).sum(axis=1) / float(pred.shape[0] - 1)
        sample_var_pred = (pred_centered ** 2).sum(axis=0) / float(pred.shape[0] - 1)
        corr_denom = np.sqrt(sample_var_ref * sample_var_pred[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            cor = np.divide(sample_cov, corr_denom, out=np.zeros_like(sample_cov), where=corr_denom != 0)
        cor = np.clip(np.nan_to_num(cor), -1.0, 1.0)
    else:
        cor = np.zeros((refs.shape[0], pred.shape[1]), dtype=np.float64)

    var_ref = np.mean(ref_centered ** 2, axis=1)
    var_pred = np.mean(pred_centered ** 2, axis=0)
    sd_ref = np.sqrt(var_ref)
    sd_pred = np.sqrt(var_pred)

    numerator = 2.0 * cor * sd_ref * sd_pred[None, :]
    denominator = var_ref + var_pred[None, :] + (ref_mean - pred_mean[None, :]) ** 2
    ccc = numerator / (denominator + 1e-8)
    return float(np.mean(ccc, axis=1).max())


def _type_metric_cache_path(predictions_dir: str) -> str:
    return os.path.join(predictions_dir, "type_metric_cache.json")


def _reference_signature(reference_seq_ids: list[str] | None, target_variant: str) -> str:
    normalized_ids = sorted(reference_seq_ids or [])
    payload = f"{target_variant}\n" + "\n".join(normalized_ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_type_metric_cache(
    predictions_dir: str,
    target_variant: str,
    reaction_annotations_dir: str,
    reference_seq_ids: list[str] | None,
) -> dict[str, dict[str, object]]:
    cache_path = _type_metric_cache_path(predictions_dir)
    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    meta = payload.get("meta")
    entries = payload.get("entries")
    if not isinstance(meta, dict) or not isinstance(entries, dict):
        return {}

    if meta.get("target_variant") != target_variant:
        return {}
    if meta.get("reaction_annotations_dir") != reaction_annotations_dir:
        return {}
    if meta.get("cache_version") != TYPE_METRIC_CACHE_VERSION:
        return {}
    if meta.get("reference_signature") != _reference_signature(reference_seq_ids, target_variant):
        return {}

    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def _save_type_metric_cache(
    predictions_dir: str,
    target_variant: str,
    reaction_annotations_dir: str,
    reference_seq_ids: list[str] | None,
    entries: dict[str, dict[str, object]],
) -> None:
    cache_path = _type_metric_cache_path(predictions_dir)
    payload = {
        "meta": {
            "target_variant": target_variant,
            "reaction_annotations_dir": reaction_annotations_dir,
            "cache_version": TYPE_METRIC_CACHE_VERSION,
            "reference_signature": _reference_signature(reference_seq_ids, target_variant),
        },
        "entries": entries,
    }
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _prediction_signature(prediction_path: str) -> str:
    stat = os.stat(prediction_path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _compute_type_metrics_for_sequence(
    seq_id: str,
    pred_seq: np.ndarray,
    appropriate_reference_sets: dict[tuple[str, str], list[np.ndarray]],
    feature_dim: int,
    reaction_annotations_dir: str,
    resampled_reference_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[int, int, float | None, float | None]:
    content_type, dominant = _annotation_bucket_key(seq_id, reaction_annotations_dir)
    if content_type is None or dominant is None:
        return 0, 0, None, None

    references = appropriate_reference_sets.get((content_type, dominant), [])
    if not references:
        return 1, 0, None, None

    bucket_key = (content_type, dominant)
    resampled_references = _get_resampled_references(
        bucket_key,
        references,
        pred_seq.shape[0],
        resampled_reference_cache,
    )
    frcorr = _max_concordance_correlation_coefficient_against_references(pred_seq, resampled_references)
    frdist = _cheap_dtw_frd_against_resampled_references(pred_seq, resampled_references, feature_dim=feature_dim)
    return 1, 1, frcorr, frdist


def _resolve_metric_target_config(
    target_variant: str,
) -> tuple[str, int, callable[[np.ndarray, np.ndarray], float]]:
    if target_variant == "full":
        return "flame_target_118", FLAME_118_DIM, _motion_frd
    if target_variant == "content":
        return "flame_target_content", FLAME_CONTENT_DIM, _content_frd
    raise ValueError(f"Unsupported target variant: {target_variant}")


@torch.no_grad()
def save_motion_predictions(
    model,
    loader,
    device,
    output_dir: str,
    use_amp: bool = False,
    eval_label: str = "val",
    log_interval: int = 1,
) -> dict[str, float]:
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    saved_sequences = 0
    saved_frames = 0
    reused_sequences = 0
    reused_frames = 0
    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        sequence_ids = [str(seq_id) for seq_id in batch["seq_ids"]]
        output_paths = [os.path.join(output_dir, f"{seq_id}.npz") for seq_id in sequence_ids]
        lengths_cpu = [int(length) for length in batch["lengths"].tolist()]
        existing_mask = [os.path.exists(output_path) for output_path in output_paths]

        if all(existing_mask):
            reused_sequences += len(sequence_ids)
            reused_frames += sum(lengths_cpu)
            if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
                prefix = _format_progress_prefix(
                    label=f"{eval_label}-predict",
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    start_time=eval_start_time,
                )
                print(
                    f"{prefix} | saved_sequences={saved_sequences} reused_sequences={reused_sequences} "
                    f"saved_frames={saved_frames} reused_frames={reused_frames}",
                    flush=True,
                )
            continue

        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)

        for sample_idx, length in enumerate(lengths.tolist()):
            seq_id = sequence_ids[sample_idx]
            pred_seq = prediction[sample_idx, :length].detach().cpu().numpy().astype(np.float32)
            output_path = output_paths[sample_idx]
            if existing_mask[sample_idx]:
                reused_sequences += 1
                reused_frames += int(length)
                continue
            np.savez_compressed(output_path, seq_id=seq_id, prediction=pred_seq)
            saved_sequences += 1
            saved_frames += int(length)

        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            prefix = _format_progress_prefix(
                label=f"{eval_label}-predict",
                batch_idx=batch_idx,
                total_batches=total_batches,
                start_time=eval_start_time,
            )
            print(
                f"{prefix} | saved_sequences={saved_sequences} reused_sequences={reused_sequences} "
                f"saved_frames={saved_frames} reused_frames={reused_frames}",
                flush=True,
            )

    return {
        "prediction_cache_sequences": float(saved_sequences),
        "prediction_cache_frames": float(saved_frames),
        "prediction_cache_reused_sequences": float(reused_sequences),
        "prediction_cache_reused_frames": float(reused_frames),
    }


def evaluate_saved_motion_metrics(
    predictions_dir: str,
    seq_ids: list[str],
    manifest: dict[str, dict[str, str]],
    target_variant: str = "full",
    reference_seq_ids: list[str] | None = None,
    eval_label: str = "val",
    log_interval: int = 1,
    reaction_annotations_dir: str = REACTION_ANNOTATIONS_DIR,
) -> dict[str, float]:
    target_key, feature_dim, _ = _resolve_metric_target_config(target_variant)

    abs_errors = []
    sq_errors = []
    frame_pred = []
    frame_target = []
    delta_pred = []
    delta_target = []
    seq_desc_pred = []
    seq_desc_target = []
    frcorr_type_list = []
    frdist_type_list = []
    evaluated_sequences = 0
    evaluated_frames = 0
    type_metric_candidates = 0
    type_metric_evaluated = 0
    type_metric_cache_hits = 0
    type_metric_cache_misses = 0

    appropriate_reference_sets: dict[tuple[str, str], list[np.ndarray]] = {}
    reference_stats = {
        "reference_candidates": 0,
        "reference_with_annotations": 0,
        "reference_with_bucket": 0,
        "reference_sequences_loaded": 0,
    }
    if reference_seq_ids:
        appropriate_reference_sets, reference_stats = _build_appropriate_reference_sets(
            reference_seq_ids=reference_seq_ids,
            manifest=manifest,
            target_key=target_key,
            annotations_dir=reaction_annotations_dir,
        )

    type_metric_cache = _load_type_metric_cache(
        predictions_dir=predictions_dir,
        target_variant=target_variant,
        reaction_annotations_dir=reaction_annotations_dir,
        reference_seq_ids=reference_seq_ids,
    )
    type_metric_cache_dirty = False
    resampled_reference_cache: dict[tuple[str, str, int], np.ndarray] = {}

    missing_predictions: list[str] = []
    total_sequences = len(seq_ids)
    eval_start_time = time.perf_counter()

    for sequence_idx, seq_id in enumerate(seq_ids, start=1):
        prediction_path = os.path.join(predictions_dir, f"{seq_id}.npz")
        if not os.path.exists(prediction_path):
            missing_predictions.append(seq_id)
            continue

        entry = manifest.get(seq_id)
        if entry is None:
            raise RuntimeError(f"Missing manifest entry for sequence {seq_id!r}")
        flame_npz = entry.get("flame_npz")
        if not flame_npz or not os.path.exists(flame_npz):
            raise RuntimeError(f"Missing FLAME target file for sequence {seq_id!r}")

        with np.load(prediction_path) as npz:
            pred_seq = np.asarray(npz["prediction"], dtype=np.float32)
        target_seq = np.asarray(flame_npz_to_targets(flame_npz)[target_key], dtype=np.float32)

        if pred_seq.ndim != 2 or pred_seq.shape[1] != feature_dim:
            raise RuntimeError(
                f"Cached prediction for {seq_id!r} has shape {pred_seq.shape}, expected (*, {feature_dim})"
            )

        if target_seq.ndim != 2 or target_seq.shape[1] != feature_dim:
            raise RuntimeError(
                f"Target sequence for {seq_id!r} has shape {target_seq.shape}, expected (*, {feature_dim})"
            )

        if pred_seq.shape[0] != target_seq.shape[0]:
            shared_length = min(pred_seq.shape[0], target_seq.shape[0])
            pred_seq = pred_seq[:shared_length]
            target_seq = target_seq[:shared_length]

        if pred_seq.size == 0:
            continue

        diff = pred_seq - target_seq
        abs_errors.append(np.abs(diff))
        sq_errors.append(diff ** 2)
        frame_pred.append(pred_seq)
        frame_target.append(target_seq)

        prediction_sig = _prediction_signature(prediction_path)
        cached_metrics = type_metric_cache.get(seq_id)
        if (
            isinstance(cached_metrics, dict)
            and cached_metrics.get("prediction_signature") == prediction_sig
            and cached_metrics.get("sequence_length") == int(pred_seq.shape[0])
        ):
            candidates = int(cached_metrics.get("type_metric_candidates", 0))
            evaluated = int(cached_metrics.get("type_metric_evaluated", 0))
            frcorr_value = cached_metrics.get("frcorr_type")
            frdist_value = cached_metrics.get("frdist_type")
            type_metric_cache_hits += 1
        else:
            candidates, evaluated, frcorr_value, frdist_value = _compute_type_metrics_for_sequence(
                seq_id=seq_id,
                pred_seq=pred_seq,
                appropriate_reference_sets=appropriate_reference_sets,
                feature_dim=feature_dim,
                reaction_annotations_dir=reaction_annotations_dir,
                resampled_reference_cache=resampled_reference_cache,
            )
            type_metric_cache[seq_id] = {
                "prediction_signature": prediction_sig,
                "sequence_length": int(pred_seq.shape[0]),
                "type_metric_candidates": candidates,
                "type_metric_evaluated": evaluated,
                "frcorr_type": frcorr_value,
                "frdist_type": frdist_value,
            }
            type_metric_cache_dirty = True
            type_metric_cache_misses += 1

        type_metric_candidates += candidates
        type_metric_evaluated += evaluated
        if evaluated:
            frcorr_type_list.append(float(frcorr_value))
            frdist_type_list.append(float(frdist_value))

        pred_delta = np.diff(pred_seq, axis=0)
        target_delta_seq = np.diff(target_seq, axis=0)
        if pred_delta.size > 0 and target_delta_seq.size > 0:
            delta_pred.append(pred_delta)
            delta_target.append(target_delta_seq)

        pred_desc = np.concatenate([
            pred_seq.mean(axis=0),
            pred_seq.std(axis=0),
            pred_delta.mean(axis=0) if pred_delta.size > 0 else np.zeros(pred_seq.shape[1], dtype=np.float32),
            pred_delta.std(axis=0) if pred_delta.size > 0 else np.zeros(pred_seq.shape[1], dtype=np.float32),
        ])
        target_desc = np.concatenate([
            target_seq.mean(axis=0),
            target_seq.std(axis=0),
            target_delta_seq.mean(axis=0) if target_delta_seq.size > 0 else np.zeros(target_seq.shape[1], dtype=np.float32),
            target_delta_seq.std(axis=0) if target_delta_seq.size > 0 else np.zeros(target_seq.shape[1], dtype=np.float32),
        ])
        seq_desc_pred.append(pred_desc.astype(np.float32))
        seq_desc_target.append(target_desc.astype(np.float32))
        evaluated_sequences += 1
        evaluated_frames += pred_seq.shape[0]

        if log_interval > 0 and (sequence_idx == 1 or sequence_idx % log_interval == 0 or sequence_idx == total_sequences):
            prefix = _format_progress_prefix(
                label=f"{eval_label}-saved-metrics",
                batch_idx=sequence_idx,
                total_batches=total_sequences,
                start_time=eval_start_time,
            )
            print(f"{prefix} | sequences={evaluated_sequences} frames={evaluated_frames}", flush=True)

    if missing_predictions:
        preview = ", ".join(missing_predictions[:5])
        suffix = "" if len(missing_predictions) <= 5 else ", ..."
        raise RuntimeError(
            f"Missing cached predictions for {len(missing_predictions)} sequences in {predictions_dir}: {preview}{suffix}"
        )

    if type_metric_cache_dirty:
        _save_type_metric_cache(
            predictions_dir=predictions_dir,
            target_variant=target_variant,
            reaction_annotations_dir=reaction_annotations_dir,
            reference_seq_ids=reference_seq_ids,
            entries=type_metric_cache,
        )

    abs_errors_arr = _stack_valid_sequences(abs_errors, feature_dim=feature_dim)
    sq_errors_arr = _stack_valid_sequences(sq_errors, feature_dim=feature_dim)
    frame_pred_arr = _stack_valid_sequences(frame_pred, feature_dim=feature_dim)
    frame_target_arr = _stack_valid_sequences(frame_target, feature_dim=feature_dim)
    delta_pred_arr = _stack_valid_sequences(delta_pred, feature_dim=feature_dim)
    delta_target_arr = _stack_valid_sequences(delta_target, feature_dim=feature_dim)

    if abs_errors_arr.shape[0] == 0:
        raise RuntimeError("No valid sequences available for metric computation")

    metrics = {
        "mae": float(abs_errors_arr.mean()),
        "rmse": float(np.sqrt(sq_errors_arr.mean())),
        "fd": _frechet_distance(frame_pred_arr, frame_target_arr),
        "frcorr_type": float(np.mean(frcorr_type_list)) if frcorr_type_list else float("nan"),
        "frdist_type": float(np.mean(frdist_type_list)) if frdist_type_list else float("nan"),
        "frcorr": float(np.mean(frcorr_type_list)) if frcorr_type_list else float("nan"),
        "frdist": float(np.mean(frdist_type_list)) if frdist_type_list else float("nan"),
    }
    metrics.update(reference_stats)
    metrics["appropriate_set_count"] = float(len(appropriate_reference_sets))
    metrics["type_metric_candidates"] = float(type_metric_candidates)
    metrics["type_metric_evaluated"] = float(type_metric_evaluated)
    metrics["type_metric_cache_hits"] = float(type_metric_cache_hits)
    metrics["type_metric_cache_misses"] = float(type_metric_cache_misses)

    for name, component_slice in flame_component_layout(feature_dim):
        metrics[f"mae_{name}"] = float(abs_errors_arr[:, component_slice].mean())
        metrics[f"rmse_{name}"] = float(np.sqrt(sq_errors_arr[:, component_slice].mean()))

    if delta_pred_arr.shape[0] > 0 and delta_target_arr.shape[0] > 0:
        metrics["fid_delta_fm"] = _frechet_distance(delta_pred_arr, delta_target_arr)
    else:
        metrics["delta_mae"] = float("nan")
        metrics["delta_rmse"] = float("nan")
        metrics["fid_delta_fm"] = float("nan")

    if seq_desc_pred and seq_desc_target:
        metrics["snd"] = _frechet_distance(np.stack(seq_desc_pred), np.stack(seq_desc_target))
    else:
        metrics["snd"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate_motion_metrics(
    model,
    loader,
    device,
    target_variant: str = "full",
    use_amp: bool = False,
    eval_label: str = "val",
    log_interval: int = 1,
    reference_seq_ids: list[str] | None = None,
    manifest: dict[str, dict[str, str]] | None = None,
    reaction_annotations_dir: str = REACTION_ANNOTATIONS_DIR,
) -> dict[str, float]:
    """Compute paired motion metrics for the evaluation split."""
    model.eval()

    abs_errors = []
    sq_errors = []
    frame_pred = []
    frame_target = []
    delta_pred = []
    delta_target = []
    seq_desc_pred = []
    seq_desc_target = []
    frcorr_type_list = []
    frdist_type_list = []
    evaluated_sequences = 0
    evaluated_frames = 0
    type_metric_candidates = 0
    type_metric_evaluated = 0
    resampled_reference_cache: dict[tuple[str, str, int], np.ndarray] = {}

    target_key, feature_dim, _ = _resolve_metric_target_config(target_variant)

    appropriate_reference_sets: dict[tuple[str, str], list[np.ndarray]] = {}
    reference_stats = {
        "reference_candidates": 0,
        "reference_with_annotations": 0,
        "reference_with_bucket": 0,
        "reference_sequences_loaded": 0,
    }
    if reference_seq_ids:
        if manifest is None:
            raise ValueError("manifest is required when reference_seq_ids are provided")
        appropriate_reference_sets, reference_stats = _build_appropriate_reference_sets(
            reference_seq_ids=reference_seq_ids,
            manifest=manifest,
            target_key=target_key,
            annotations_dir=reaction_annotations_dir,
        )

    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target = batch[target_key].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)

        for sample_idx, length in enumerate(lengths.tolist()):
            seq_id = str(batch["seq_ids"][sample_idx])
            pred_seq = prediction[sample_idx, :length].detach().cpu().numpy().astype(np.float32)
            target_seq = target[sample_idx, :length].detach().cpu().numpy().astype(np.float32)
            if pred_seq.size == 0:
                continue

            diff = pred_seq - target_seq
            abs_errors.append(np.abs(diff))
            sq_errors.append(diff ** 2)
            frame_pred.append(pred_seq)
            frame_target.append(target_seq)

            candidates, evaluated, frcorr_value, frdist_value = _compute_type_metrics_for_sequence(
                seq_id=seq_id,
                pred_seq=pred_seq,
                appropriate_reference_sets=appropriate_reference_sets,
                feature_dim=feature_dim,
                reaction_annotations_dir=reaction_annotations_dir,
                resampled_reference_cache=resampled_reference_cache,
            )
            type_metric_candidates += candidates
            type_metric_evaluated += evaluated
            if evaluated:
                frcorr_type_list.append(float(frcorr_value))
                frdist_type_list.append(float(frdist_value))

            pred_delta = np.diff(pred_seq, axis=0)
            target_delta_seq = np.diff(target_seq, axis=0)
            if pred_delta.size > 0 and target_delta_seq.size > 0:
                delta_pred.append(pred_delta)
                delta_target.append(target_delta_seq)

            pred_desc = np.concatenate([
                pred_seq.mean(axis=0),
                pred_seq.std(axis=0),
                pred_delta.mean(axis=0) if pred_delta.size > 0 else np.zeros(pred_seq.shape[1], dtype=np.float32),
                pred_delta.std(axis=0) if pred_delta.size > 0 else np.zeros(pred_seq.shape[1], dtype=np.float32),
            ])
            target_desc = np.concatenate([
                target_seq.mean(axis=0),
                target_seq.std(axis=0),
                target_delta_seq.mean(axis=0) if target_delta_seq.size > 0 else np.zeros(target_seq.shape[1], dtype=np.float32),
                target_delta_seq.std(axis=0) if target_delta_seq.size > 0 else np.zeros(target_seq.shape[1], dtype=np.float32),
            ])
            seq_desc_pred.append(pred_desc.astype(np.float32))
            seq_desc_target.append(target_desc.astype(np.float32))
            evaluated_sequences += 1
            evaluated_frames += length

        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            prefix = _format_progress_prefix(
                label=f"{eval_label}-metrics",
                batch_idx=batch_idx,
                total_batches=total_batches,
                start_time=eval_start_time,
            )
            print(f"{prefix} | sequences={evaluated_sequences} frames={evaluated_frames}", flush=True)

    abs_errors_arr = _stack_valid_sequences(abs_errors, feature_dim=feature_dim)
    sq_errors_arr = _stack_valid_sequences(sq_errors, feature_dim=feature_dim)
    frame_pred_arr = _stack_valid_sequences(frame_pred, feature_dim=feature_dim)
    frame_target_arr = _stack_valid_sequences(frame_target, feature_dim=feature_dim)
    delta_pred_arr = _stack_valid_sequences(delta_pred, feature_dim=feature_dim)
    delta_target_arr = _stack_valid_sequences(delta_target, feature_dim=feature_dim)

    if abs_errors_arr.shape[0] == 0:
        raise RuntimeError("No valid sequences available for metric computation")

    metrics = {
        "mae": float(abs_errors_arr.mean()),
        "rmse": float(np.sqrt(sq_errors_arr.mean())),
        "fd": _frechet_distance(frame_pred_arr, frame_target_arr),
        "frcorr_type": float(np.mean(frcorr_type_list)) if frcorr_type_list else float("nan"),
        "frdist_type": float(np.mean(frdist_type_list)) if frdist_type_list else float("nan"),
        "frcorr": float(np.mean(frcorr_type_list)) if frcorr_type_list else float("nan"),
        "frdist": float(np.mean(frdist_type_list)) if frdist_type_list else float("nan"),
    }
    metrics.update(reference_stats)
    metrics["appropriate_set_count"] = float(len(appropriate_reference_sets))
    metrics["type_metric_candidates"] = float(type_metric_candidates)
    metrics["type_metric_evaluated"] = float(type_metric_evaluated)

    for name, component_slice in flame_component_layout(feature_dim):
        metrics[f"mae_{name}"] = float(abs_errors_arr[:, component_slice].mean())
        metrics[f"rmse_{name}"] = float(np.sqrt(sq_errors_arr[:, component_slice].mean()))

    if delta_pred_arr.shape[0] > 0 and delta_target_arr.shape[0] > 0:
        delta_abs = np.abs(delta_pred_arr - delta_target_arr)
        delta_sq = (delta_pred_arr - delta_target_arr) ** 2
        # metrics["delta_mae"] = float(delta_abs.mean())
        # metrics["delta_rmse"] = float(np.sqrt(delta_sq.mean()))
        metrics["fid_delta_fm"] = _frechet_distance(delta_pred_arr, delta_target_arr)
    else:
        metrics["delta_mae"] = float("nan")
        metrics["delta_rmse"] = float("nan")
        metrics["fid_delta_fm"] = float("nan")

    if seq_desc_pred and seq_desc_target:
        metrics["snd"] = _frechet_distance(np.stack(seq_desc_pred), np.stack(seq_desc_target))
    else:
        metrics["snd"] = float("nan")

    return metrics


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    extra_state: dict[str, object] | None = None,
) -> None:
    """Save a checkpoint for benchmark training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | str) -> dict[str, object]:
    """Load a checkpoint file while remaining compatible with older torch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint payload in {path!r}: expected a dict")
    return checkpoint


def checkpoint_state_dict(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    """Extract model weights from either a full training checkpoint or a raw state dict."""
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        filtered_state_dict = dict(state_dict)
        filtered_state_dict.pop("indices", None)
        return filtered_state_dict

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        filtered_state_dict = dict(checkpoint)
        filtered_state_dict.pop("indices", None)
        return filtered_state_dict  # type: ignore[return-value]

    raise RuntimeError("Checkpoint does not contain a model state dict")


def extract_checkpoint_metric(checkpoint: dict[str, object], metric_name: str = "loss_total") -> float | None:
    """Read the primary validation metric from a checkpoint when available."""
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)

    value = checkpoint.get("val_loss")
    if isinstance(value, (int, float)):
        return float(value)

    return None


def resume_training_state(
    path: str,
    model: nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, object]:
    """Restore model weights and, when available, optimizer/scaler state for training resume."""
    checkpoint = load_checkpoint(path, device)
    model.load_state_dict(checkpoint_state_dict(checkpoint))

    restored_optimizer = False
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        restored_optimizer = True

    restored_scaler = False
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
        restored_scaler = True

    epoch = checkpoint.get("epoch")
    start_epoch = int(epoch) if isinstance(epoch, int) else 0

    return {
        "checkpoint": checkpoint,
        "epoch": start_epoch,
        "primary_metric": extract_checkpoint_metric(checkpoint),
        "restored_optimizer": restored_optimizer,
        "restored_scaler": restored_scaler,
        "is_full_checkpoint": "model_state_dict" in checkpoint,
    }