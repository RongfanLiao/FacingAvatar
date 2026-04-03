"""LookingFace-compatible motion-only port of the baseline motion_transvae model."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM
from config import VIDEO_CANVAS_SIZE, WAV2VEC_DIM


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
    """Transformer decoder that predicts reduced 58-d motion targets."""

    def __init__(self, output_dim: int = FLAME_58_DIM, feature_dim: int = 128, n_head: int = 4, max_seq_len: int = 1024):
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

    def __init__(self, feature_dim: int = 128, chunk_size: int = 8):
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


class MotionOnlyTransformerVAE(nn.Module):
    """Motion-only port of baseline motion_transvae for LookingFace benchmarking.

    Uses BaselineSpeakerEncoder (Conv3D video + wav2vec audio, concat+linear fusion)
    following the baseline_react2025 architecture.
    """

    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        output_dim: int = FLAME_CONTENT_DIM,
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
    """Masked reconstruction plus KL loss for motion target variants."""

    kl_p: float = 1e-5
    w_exp: float = 2.0
    w_jaw: float = 2.0
    w_neck: float = 2.0
    w_eyes: float = 2.0
    w_rot: float = 4.0
    w_tran: float = 4.0
    target_variant: str = "content"

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
            rec_loss = (
                self.w_exp * exp_loss
                + self.w_jaw * jaw_loss
                + self.w_neck * neck_loss
                + self.w_eyes * eyes_loss
            )
            component_logs = {
                "loss_exp": float(exp_loss.item()),
                "loss_jaw": float(jaw_loss.item()),
                "loss_neck": float(neck_loss.item()),
                "loss_eyes": float(eyes_loss.item()),
            }
        else:
            raise ValueError(f"Unsupported target_variant: {self.target_variant}")

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


def train_motion_transvae(
    model, loader, optimizer, criterion, device, div_p: float = 10.0, scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """One training epoch for the LookingFace motion-only TransVAE port."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    use_amp = scaler is not None

    for batch in loader:
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target_key = "flame_target_58" if criterion.target_variant == "motion58" else "flame_target_content"
        target = batch[target_key].to(device)
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

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_motion_transvae(model, loader, criterion, device, use_amp: bool = False) -> dict[str, float]:
    """Validation loop for the LookingFace motion-only TransVAE port."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target_key = "flame_target_58" if criterion.target_variant == "motion58" else "flame_target_content"
        target = batch[target_key].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, distribution = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)
            _, logs = criterion(prediction, target, distribution, padding_mask)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

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
    """Motion-space FRD adapted from the baseline grouped DTW metric."""
    groups = [
        (0, 52, 1.0 / 52.0),
        (52, 55, 1.0 / 3.0),
        (55, 58, 1.0 / 3.0),
    ]
    total = 0.0
    for start, end, weight in groups:
        total += weight * _dtw_distance(pred_seq[:, start:end].astype(np.float32), target_seq[:, start:end].astype(np.float32))
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
        total += weight * _dtw_distance(pred_seq[:, start:end].astype(np.float32), target_seq[:, start:end].astype(np.float32))
    return float(total)


def _stack_valid_sequences(items: Iterable[np.ndarray], feature_dim: int) -> np.ndarray:
    arrays = [item for item in items if item.size > 0]
    if not arrays:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


@torch.no_grad()
def evaluate_motion_metrics(model, loader, device, target_variant: str = "content", use_amp: bool = False) -> dict[str, float]:
    """Compute paired motion metrics for the validation split."""
    model.eval()

    abs_errors = []
    sq_errors = []
    delta_pred = []
    delta_target = []
    seq_desc_pred = []
    seq_desc_target = []
    frc_list = []
    frd_list = []

    if target_variant == "motion58":
        target_key = "flame_target_58"
        feature_dim = FLAME_58_DIM
        frd_func = _motion_frd
    elif target_variant == "content":
        target_key = "flame_target_content"
        feature_dim = FLAME_CONTENT_DIM
        frd_func = _content_frd
    else:
        raise ValueError(f"Unsupported target_variant: {target_variant}")

    for batch in loader:
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch.get("left_video_frames", batch.get("left_video_feat")).to(device)
        target = batch[target_key].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction, _ = model(left_audio, left_video, lengths=lengths, padding_mask=padding_mask)

        for sample_idx, length in enumerate(lengths.tolist()):
            pred_seq = prediction[sample_idx, :length].detach().cpu().numpy().astype(np.float32)
            target_seq = target[sample_idx, :length].detach().cpu().numpy().astype(np.float32)
            if pred_seq.size == 0:
                continue

            diff = pred_seq - target_seq
            abs_errors.append(np.abs(diff))
            sq_errors.append(diff ** 2)
            frc_list.append(_concordance_correlation_coefficient(target_seq, pred_seq))
            frd_list.append(frd_func(pred_seq, target_seq))

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

    abs_errors_arr = _stack_valid_sequences(abs_errors, feature_dim=feature_dim)
    sq_errors_arr = _stack_valid_sequences(sq_errors, feature_dim=feature_dim)
    delta_pred_arr = _stack_valid_sequences(delta_pred, feature_dim=feature_dim)
    delta_target_arr = _stack_valid_sequences(delta_target, feature_dim=feature_dim)

    if abs_errors_arr.shape[0] == 0:
        raise RuntimeError("No valid sequences available for metric computation")

    metrics = {
        "mae": float(abs_errors_arr.mean()),
        "rmse": float(np.sqrt(sq_errors_arr.mean())),
        "frcorr": float(np.mean(frc_list)),
        "frdist": float(np.mean(frd_list)),
    }

    if target_variant == "motion58":
        metrics.update({
            "mae_expr": float(abs_errors_arr[:, :52].mean()),
            "mae_rot": float(abs_errors_arr[:, 52:55].mean()),
            "mae_tran": float(abs_errors_arr[:, 55:58].mean()),
            "rmse_expr": float(np.sqrt(sq_errors_arr[:, :52].mean())),
            "rmse_rot": float(np.sqrt(sq_errors_arr[:, 52:55].mean())),
            "rmse_tran": float(np.sqrt(sq_errors_arr[:, 55:58].mean())),
        })
    else:
        metrics.update({
            "mae_expr": float(abs_errors_arr[:, :100].mean()),
            "mae_jaw": float(abs_errors_arr[:, 100:103].mean()),
            "mae_neck": float(abs_errors_arr[:, 103:106].mean()),
            "mae_eyes": float(abs_errors_arr[:, 106:112].mean()),
            "rmse_expr": float(np.sqrt(sq_errors_arr[:, :100].mean())),
            "rmse_jaw": float(np.sqrt(sq_errors_arr[:, 100:103].mean())),
            "rmse_neck": float(np.sqrt(sq_errors_arr[:, 103:106].mean())),
            "rmse_eyes": float(np.sqrt(sq_errors_arr[:, 106:112].mean())),
        })

    if delta_pred_arr.shape[0] > 0 and delta_target_arr.shape[0] > 0:
        delta_abs = np.abs(delta_pred_arr - delta_target_arr)
        delta_sq = (delta_pred_arr - delta_target_arr) ** 2
        metrics["delta_mae"] = float(delta_abs.mean())
        metrics["delta_rmse"] = float(np.sqrt(delta_sq.mean()))
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


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    """Save a checkpoint for benchmark training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )