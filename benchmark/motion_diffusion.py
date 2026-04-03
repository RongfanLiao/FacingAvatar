"""LookingFace-compatible motion_diffusion port with baseline-style conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from benchmark.motion_transvae import PositionalEncoding, evaluate_motion_metrics
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def build_inference_indices(train_timesteps: int, inference_timesteps: int, spacing: str) -> np.ndarray:
    """Build spaced diffusion indices following the baseline scheduler modes."""
    inference_timesteps = min(inference_timesteps, train_timesteps)
    if spacing == "linspace":
        indices = np.linspace(0, train_timesteps - 1, inference_timesteps).round()[::-1]
    elif spacing == "leading":
        step_ratio = max(train_timesteps // inference_timesteps, 1)
        indices = (np.arange(0, inference_timesteps) * step_ratio).round()[::-1]
    elif spacing == "trailing":
        step_ratio = max(train_timesteps // inference_timesteps, 1)
        indices = np.round(np.arange(train_timesteps, 0, -step_ratio))
    else:
        indices = np.arange(train_timesteps - 1, -1, -1)
    indices = np.clip(indices.astype(np.int64), 0, train_timesteps - 1)
    return indices


class DiffusionDenoiser(nn.Module):
    """Baseline-shaped transformer decoder with classifier-free condition dropout."""

    def __init__(
        self,
        target_dim: int,
        feature_dim: int,
        audio_dim: int,
        video_dim: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        guidance_scale: float = 1.0,
        audio_drop_prob: float = 0.2,
        video_drop_prob: float = 0.2,
        latent_drop_prob: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.guidance_scale = guidance_scale
        self.audio_drop_prob = audio_drop_prob
        self.video_drop_prob = video_drop_prob
        self.latent_drop_prob = latent_drop_prob

        self.target_proj = nn.Linear(target_dim, feature_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.audio_proj = nn.Linear(audio_dim, feature_dim)
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, num_layers))

        self.query_pos = PositionalEncoding(feature_dim, dropout=dropout)
        self.mem_pos = PositionalEncoding(feature_dim, dropout=dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, target_dim),
        )

    def _mask_cond(self, feature: torch.Tensor, drop_prob: float, mode: str) -> torch.Tensor:
        if feature.shape[1] == 0:
            return feature
        if mode == "test":
            uncond, cond = feature.chunk(2, dim=0)
            return torch.cat([torch.zeros_like(uncond), cond], dim=0)
        if drop_prob <= 0.0:
            return feature
        keep_mask = 1.0 - torch.bernoulli(
            torch.full((feature.shape[0], 1, 1), drop_prob, device=feature.device, dtype=feature.dtype)
        )
        return feature * keep_mask

    def _encode_conditions(
        self,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        audio_tokens = self.audio_proj(left_audio_feat)
        audio_tokens = self.query_pos(audio_tokens)
        audio_tokens = self.audio_encoder(audio_tokens, src_key_padding_mask=padding_mask)
        audio_tokens = self._mask_cond(audio_tokens, self.audio_drop_prob, mode)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled_audio = (audio_tokens * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_audio = audio_tokens.mean(dim=1)

        latent_token = self.latent_proj(pooled_audio).unsqueeze(1)
        latent_token = self._mask_cond(latent_token, self.latent_drop_prob, mode)

        video_token = self.video_proj(left_video_feat).unsqueeze(1)
        video_token = self._mask_cond(video_token, self.video_drop_prob, mode)

        memory = torch.cat([latent_token, video_token, audio_tokens], dim=1)
        memory = self.mem_pos(memory)

        memory_key_padding_mask = None
        if padding_mask is not None:
            token_mask = torch.zeros(padding_mask.shape[0], 2, dtype=torch.bool, device=padding_mask.device)
            memory_key_padding_mask = torch.cat([token_mask, padding_mask], dim=1)

        return memory, memory_key_padding_mask

    def _forward_impl(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
        mode: str,
    ) -> torch.Tensor:
        target_tokens = self.target_proj(noisy_target)
        target_tokens = self.query_pos(target_tokens)

        time_token = self.time_proj(timestep_embedding(timesteps, self.feature_dim)).unsqueeze(1)
        memory, memory_key_padding_mask = self._encode_conditions(
            left_audio_feat=left_audio_feat,
            left_video_feat=left_video_feat,
            padding_mask=padding_mask,
            mode=mode,
        )
        memory = torch.cat([time_token, memory], dim=1)

        if memory_key_padding_mask is not None:
            time_mask = torch.zeros(memory_key_padding_mask.shape[0], 1, dtype=torch.bool, device=memory_key_padding_mask.device)
            memory_key_padding_mask = torch.cat([time_mask, memory_key_padding_mask], dim=1)

        decoded = self.decoder(
            tgt=target_tokens,
            memory=memory,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_head(decoded)

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._forward_impl(noisy_target, timesteps, left_audio_feat, left_video_feat, padding_mask, mode="train")

    def forward_with_cond_scale(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.guidance_scale == 1.0:
            return self._forward_impl(noisy_target, timesteps, left_audio_feat, left_video_feat, padding_mask, mode="train")

        model_input = torch.cat([noisy_target, noisy_target], dim=0)
        time_input = torch.cat([timesteps, timesteps], dim=0)
        audio_input = torch.cat([left_audio_feat, left_audio_feat], dim=0)
        video_input = torch.cat([left_video_feat, left_video_feat], dim=0)
        if padding_mask is not None:
            padding_input = torch.cat([padding_mask, padding_mask], dim=0)
        else:
            padding_input = None

        prediction = self._forward_impl(model_input, time_input, audio_input, video_input, padding_input, mode="test")
        pred_uncond, pred_cond = prediction.chunk(2, dim=0)
        return pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)


class MotionDiffusionModel(nn.Module):
    """DDIM-style motion diffusion model adapted to LookingFace content targets."""

    def __init__(
        self,
        audio_dim: int = 1280,
        video_dim: int = 3584,
        target_variant: str = "content",
        feature_dim: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        train_timesteps: int = 1000,
        inference_timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_sample: float = 5.0,
        guidance_scale: float = 1.0,
        audio_drop_prob: float = 0.2,
        video_drop_prob: float = 0.2,
        latent_drop_prob: float = 0.2,
        timestep_spacing: str = "leading",
        ddim_eta: float = 0.0,
    ):
        super().__init__()
        if target_variant == "content":
            self.target_dim = FLAME_CONTENT_DIM
        elif target_variant == "motion58":
            self.target_dim = FLAME_58_DIM
        else:
            raise ValueError(f"Unsupported target_variant: {target_variant}")

        self.target_variant = target_variant
        self.train_timesteps = train_timesteps
        self.inference_timesteps = min(inference_timesteps, train_timesteps)
        self.clip_sample = clip_sample
        self.ddim_eta = ddim_eta
        self.denoiser = DiffusionDenoiser(
            target_dim=self.target_dim,
            feature_dim=feature_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            guidance_scale=guidance_scale,
            audio_drop_prob=audio_drop_prob,
            video_drop_prob=video_drop_prob,
            latent_drop_prob=latent_drop_prob,
        )

        betas = torch.linspace(beta_start, beta_end, train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("indices", torch.from_numpy(build_inference_indices(train_timesteps, self.inference_timesteps, timestep_spacing)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale_clean = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return scale_clean * x_start + scale_noise * noise

    def predict_x0(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
        use_guidance: bool,
    ) -> torch.Tensor:
        if use_guidance:
            prediction = self.denoiser.forward_with_cond_scale(
                noisy_target=noisy_target,
                timesteps=timesteps,
                left_audio_feat=left_audio_feat,
                left_video_feat=left_video_feat,
                padding_mask=padding_mask,
            )
        else:
            prediction = self.denoiser(
                noisy_target=noisy_target,
                timesteps=timesteps,
                left_audio_feat=left_audio_feat,
                left_video_feat=left_video_feat,
                padding_mask=padding_mask,
            )
        return prediction.clamp(min=-self.clip_sample, max=self.clip_sample)

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del lengths
        if target is None:
            return self.sample(left_audio_feat, left_video_feat, padding_mask), {}

        batch_size = target.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.train_timesteps, (batch_size,), device=target.device)
        if noise is None:
            noise = torch.randn_like(target)

        noisy_target = self.q_sample(target, timesteps, noise)
        if padding_mask is not None:
            noisy_target = noisy_target.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        prediction = self.predict_x0(
            noisy_target=noisy_target,
            timesteps=timesteps,
            left_audio_feat=left_audio_feat,
            left_video_feat=left_video_feat,
            padding_mask=padding_mask,
            use_guidance=False,
        )
        if padding_mask is not None:
            prediction = prediction.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return prediction, {"timesteps": timesteps, "noise": noise, "noisy_target": noisy_target}

    @torch.no_grad()
    def sample(
        self,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = left_audio_feat.shape[:2]
        sample = torch.randn(batch_size, seq_len, self.target_dim, device=left_audio_feat.device)
        if padding_mask is not None:
            sample = sample.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        step_indices = self.indices.to(left_audio_feat.device)
        for idx, step in enumerate(step_indices):
            t = torch.full((batch_size,), int(step.item()), device=left_audio_feat.device, dtype=torch.long)
            t_prev_value = int(step_indices[idx + 1].item()) if idx < len(step_indices) - 1 else 0
            t_prev = torch.full((batch_size,), t_prev_value, device=left_audio_feat.device, dtype=torch.long)

            pred_x0 = self.predict_x0(
                noisy_target=sample,
                timesteps=t,
                left_audio_feat=left_audio_feat,
                left_video_feat=left_video_feat,
                padding_mask=padding_mask,
                use_guidance=True,
            )

            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
            alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1)
            eps = (sample - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt((1.0 - alpha_bar_t).clamp_min(1e-8))
            sigma = (
                self.ddim_eta
                * torch.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).clamp_min(0.0))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev).clamp_min(0.0))
            )
            mean_pred = pred_x0 * torch.sqrt(alpha_bar_prev) + torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp_min(0.0)) * eps
            nonzero_mask = (t != 0).float().view(-1, 1, 1)
            noise = torch.randn_like(sample)
            sample = mean_pred + nonzero_mask * sigma * noise

            if padding_mask is not None:
                sample = sample.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return sample.clamp(min=-self.clip_sample, max=self.clip_sample)

    def get_model_name(self) -> str:
        return "MotionDiffusionModel"


@dataclass
class MotionDiffusionLoss:
    """Masked content-aware reconstruction loss for diffusion training."""

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
        sq_error = (prediction - target) ** 2

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            denom = valid_mask.sum().clamp_min(1.0) * value.shape[-1]
            return (value * valid_mask).sum() / denom

        if self.target_variant == "motion58":
            exp_loss = masked_mean(sq_error[:, :, :52])
            rot_loss = masked_mean(sq_error[:, :, 52:55])
            tran_loss = masked_mean(sq_error[:, :, 55:58])
            rec_loss = self.w_exp * exp_loss + self.w_rot * rot_loss + self.w_tran * tran_loss
            component_logs = {
                "loss_exp": float(exp_loss.item()),
                "loss_rot": float(rot_loss.item()),
                "loss_tran": float(tran_loss.item()),
            }
        elif self.target_variant == "content":
            exp_loss = masked_mean(sq_error[:, :, :100])
            jaw_loss = masked_mean(sq_error[:, :, 100:103])
            neck_loss = masked_mean(sq_error[:, :, 103:106])
            eyes_loss = masked_mean(sq_error[:, :, 106:112])
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


def train_motion_diffusion(
    model: MotionDiffusionModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: MotionDiffusionLoss,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Run one diffusion training epoch."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_feat"].to(device)
        target = _resolve_target(batch, criterion.target_variant).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()
        prediction, aux = model(
            left_audio_feat=left_audio,
            left_video_feat=left_video,
            lengths=lengths,
            padding_mask=padding_mask,
            target=target,
        )
        loss, logs = criterion(prediction, target, padding_mask)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        logs["loss_total_with_clip"] = float(loss.item())
        logs["mean_timestep"] = float(aux["timesteps"].float().mean().item())
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_motion_diffusion(
    model: MotionDiffusionModel,
    loader,
    criterion: MotionDiffusionLoss,
    device: torch.device,
) -> dict[str, float]:
    """Validation loop using a single noisy denoising step objective."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_feat"].to(device)
        target = _resolve_target(batch, criterion.target_variant).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        prediction, aux = model(
            left_audio_feat=left_audio,
            left_video_feat=left_video,
            lengths=lengths,
            padding_mask=padding_mask,
            target=target,
        )
        _, logs = criterion(prediction, target, padding_mask)
        logs["mean_timestep"] = float(aux["timesteps"].float().mean().item())
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_motion_diffusion_metrics(
    model: MotionDiffusionModel,
    loader,
    device: torch.device,
    target_variant: str = "content",
) -> dict[str, float]:
    """Run the shared benchmark metric stack on sampled diffusion outputs."""

    class _SamplerWrapper(nn.Module):
        def __init__(self, inner: MotionDiffusionModel):
            super().__init__()
            self.inner = inner

        def forward(self, left_audio_feat, left_video_feat, lengths, padding_mask=None):
            del lengths
            return self.inner.sample(left_audio_feat, left_video_feat, padding_mask), None

    return evaluate_motion_metrics(_SamplerWrapper(model), loader, device=device, target_variant=target_variant)