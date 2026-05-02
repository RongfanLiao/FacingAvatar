"""LookingFace-compatible motion flow matching port with baseline-style conditioning."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn

from benchmark.motion_diffusion import timestep_embedding
from benchmark.motion_transvae import PositionalEncoding, VideoEncoder, evaluate_motion_metrics
from benchmark.targets import FLAME_118_DIM, flame_component_layout, flame_target_key, flame_target_variant
from config import WAV2VEC_DIM


def _format_progress_logs(logs: dict[str, float], prefix: str) -> str:
    ordered_keys = [
        "loss_total",
        "loss_flow",
        "loss_rec",
        "loss_vel",
        "loss_flow_exp",
        "loss_flow_jaw",
        "loss_flow_rot",
        "loss_flow_neck",
        "loss_flow_eyes",
        "loss_flow_tran",
        "mean_time",
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


class FlowMatchingVelocityField(nn.Module):
    """Conditional velocity-field regressor for continuous-time motion generation."""

    def __init__(
        self,
        target_dim: int,
        feature_dim: int,
        audio_dim: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        video_chunk_size: int = 32,
        guidance_scale: float = 1.0,
        audio_drop_prob: float = 0.2,
        video_drop_prob: float = 0.2,
        latent_drop_prob: float = 0.2,
        time_embed_scale: float = 1000.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.guidance_scale = guidance_scale
        self.audio_drop_prob = audio_drop_prob
        self.video_drop_prob = video_drop_prob
        self.latent_drop_prob = latent_drop_prob
        self.time_embed_scale = time_embed_scale

        self.target_proj = nn.Linear(target_dim, feature_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.video_encoder = VideoEncoder(feature_dim=feature_dim, chunk_size=video_chunk_size)
        self.audio_proj = nn.Linear(audio_dim, feature_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
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
        if mode != "train":
            return feature
        if drop_prob <= 0.0:
            return feature
        keep_mask = 1.0 - torch.bernoulli(
            torch.full((feature.shape[0], 1, 1), drop_prob, device=feature.device, dtype=feature.dtype)
        )
        return feature * keep_mask

    def _encode_conditions(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        audio_tokens = self.audio_proj(left_audio_feat)
        audio_tokens = self.query_pos(audio_tokens)
        audio_tokens = self.audio_encoder(audio_tokens, src_key_padding_mask=padding_mask)
        audio_tokens = self._mask_cond(audio_tokens, self.audio_drop_prob, mode)

        video_tokens = self.video_encoder(left_video_frames)
        video_tokens = self.query_pos(video_tokens)
        video_tokens = self._mask_cond(video_tokens, self.video_drop_prob, mode)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled_audio = (audio_tokens * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
            pooled_video = (video_tokens * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_audio = audio_tokens.mean(dim=1)
            pooled_video = video_tokens.mean(dim=1)

        latent_token = self.latent_proj(torch.cat([pooled_audio, pooled_video], dim=-1)).unsqueeze(1)
        latent_token = self._mask_cond(latent_token, self.latent_drop_prob, mode)

        memory = torch.cat([latent_token, video_tokens, audio_tokens], dim=1)
        memory = self.mem_pos(memory)

        memory_key_padding_mask = None
        if padding_mask is not None:
            token_mask = torch.zeros(padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device)
            memory_key_padding_mask = torch.cat([token_mask, padding_mask, padding_mask], dim=1)

        return memory, memory_key_padding_mask

    def _forward_impl(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
        mode: str,
    ) -> torch.Tensor:
        target_tokens = self.target_proj(x_t)
        target_tokens = self.query_pos(target_tokens)

        time_values = times.float() * self.time_embed_scale
        time_token = self.time_proj(timestep_embedding(time_values, self.feature_dim)).unsqueeze(1)
        memory, memory_key_padding_mask = self._encode_conditions(
            left_audio_feat=left_audio_feat,
            left_video_frames=left_video_frames,
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
        x_t: torch.Tensor,
        times: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._forward_impl(x_t, times, left_audio_feat, left_video_frames, padding_mask, mode="train")

    def forward_with_cond_scale(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.guidance_scale == 1.0:
            return self._forward_impl(x_t, times, left_audio_feat, left_video_frames, padding_mask, mode="sample")

        model_input = torch.cat([x_t, x_t], dim=0)
        time_input = torch.cat([times, times], dim=0)
        audio_input = torch.cat([left_audio_feat, left_audio_feat], dim=0)
        video_input = torch.cat([left_video_frames, left_video_frames], dim=0)
        if padding_mask is not None:
            padding_input = torch.cat([padding_mask, padding_mask], dim=0)
        else:
            padding_input = None

        prediction = self._forward_impl(model_input, time_input, audio_input, video_input, padding_input, mode="test")
        pred_uncond, pred_cond = prediction.chunk(2, dim=0)
        return pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)


class MotionFlowMatchingModel(nn.Module):
    """Continuous-time conditional flow matching model for LookingFace motion generation."""

    def __init__(
        self,
        audio_dim: int = WAV2VEC_DIM,
        target_dim: int = FLAME_118_DIM,
        feature_dim: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        video_chunk_size: int = 32,
        solver: str = "euler",
        solver_steps: int = 50,
        clip_sample: float = 5.0,
        guidance_scale: float = 1.0,
        audio_drop_prob: float = 0.2,
        video_drop_prob: float = 0.2,
        latent_drop_prob: float = 0.2,
        time_embed_scale: float = 1000.0,
        time_sampling: str = "uniform",
        time_beta_alpha: float = 1.0,
        time_beta_beta: float = 1.0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.solver = solver
        self.solver_steps = max(int(solver_steps), 1)
        self.clip_sample = clip_sample
        self.time_sampling = time_sampling
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.velocity_field = FlowMatchingVelocityField(
            target_dim=self.target_dim,
            feature_dim=feature_dim,
            audio_dim=audio_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            video_chunk_size=video_chunk_size,
            guidance_scale=guidance_scale,
            audio_drop_prob=audio_drop_prob,
            video_drop_prob=video_drop_prob,
            latent_drop_prob=latent_drop_prob,
            time_embed_scale=time_embed_scale,
        )

    def _sample_times(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "beta":
            distribution = torch.distributions.Beta(self.time_beta_alpha, self.time_beta_beta)
            return distribution.sample((batch_size,)).to(device=device, dtype=torch.float32)
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def interpolate_path(self, source: torch.Tensor, target: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_view = times.view(-1, 1, 1)
        x_t = (1.0 - time_view) * source + time_view * target
        target_velocity = target - source
        return x_t, target_velocity

    def reconstruct_target(self, x_t: torch.Tensor, pred_velocity: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        time_view = times.view(-1, 1, 1)
        return x_t + (1.0 - time_view) * pred_velocity

    def forward(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del lengths
        if target is None:
            return self.sample(left_audio_feat, left_video_frames, padding_mask), {}

        batch_size = target.shape[0]
        if times is None:
            times = self._sample_times(batch_size, target.device)
        if source is None:
            source = torch.randn_like(target)

        x_t, target_velocity = self.interpolate_path(source, target, times)
        if padding_mask is not None:
            x_t = x_t.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            target_velocity = target_velocity.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        pred_velocity = self.velocity_field(
            x_t=x_t,
            times=times,
            left_audio_feat=left_audio_feat,
            left_video_frames=left_video_frames,
            padding_mask=padding_mask,
        )
        if padding_mask is not None:
            pred_velocity = pred_velocity.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        pred_target = self.reconstruct_target(x_t, pred_velocity, times)
        if padding_mask is not None:
            pred_target = pred_target.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pred_target = pred_target.clamp(min=-self.clip_sample, max=self.clip_sample)

        return pred_velocity, {
            "times": times,
            "source": source,
            "x_t": x_t,
            "target_velocity": target_velocity,
            "pred_target": pred_target,
        }

    @torch.no_grad()
    def sample(
        self,
        left_audio_feat: torch.Tensor,
        left_video_frames: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = left_audio_feat.shape[:2]
        sample = torch.randn(batch_size, seq_len, self.target_dim, device=left_audio_feat.device)
        if padding_mask is not None:
            sample = sample.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        time_grid = torch.linspace(0.0, 1.0, self.solver_steps + 1, device=left_audio_feat.device, dtype=torch.float32)
        for step_idx in range(self.solver_steps):
            t_value = time_grid[step_idx]
            t_next = time_grid[step_idx + 1]
            dt = float((t_next - t_value).item())
            times = torch.full((batch_size,), float(t_value.item()), device=left_audio_feat.device, dtype=torch.float32)

            velocity = self.velocity_field.forward_with_cond_scale(
                x_t=sample,
                times=times,
                left_audio_feat=left_audio_feat,
                left_video_frames=left_video_frames,
                padding_mask=padding_mask,
            )

            if self.solver == "heun" and step_idx < self.solver_steps - 1:
                predictor = sample + dt * velocity
                if padding_mask is not None:
                    predictor = predictor.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                next_times = torch.full((batch_size,), float(t_next.item()), device=left_audio_feat.device, dtype=torch.float32)
                velocity_next = self.velocity_field.forward_with_cond_scale(
                    x_t=predictor,
                    times=next_times,
                    left_audio_feat=left_audio_feat,
                    left_video_frames=left_video_frames,
                    padding_mask=padding_mask,
                )
                sample = sample + 0.5 * dt * (velocity + velocity_next)
            else:
                sample = sample + dt * velocity

            sample = sample.clamp(min=-self.clip_sample, max=self.clip_sample)
            if padding_mask is not None:
                sample = sample.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return sample

    def get_model_name(self) -> str:
        return "MotionFlowMatchingModel"


@dataclass
class FlowMatchingLoss:
    """Flow-matching supervision with optional reconstruction diagnostics and regularizers."""

    flow_weight: float = 1.0
    reconstruction_weight: float = 0.0
    velocity_weight: float = 0.0
    w_exp: float = 2.0
    w_jaw: float = 2.0
    w_rot: float = 1.0
    w_neck: float = 2.0
    w_eyes: float = 2.0
    w_tran: float = 0.1

    def _masked_mean(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        denom = valid_mask.sum().clamp_min(1.0) * value.shape[-1]
        return (value * valid_mask).sum() / denom

    def _weighted_component_loss(
        self,
        sq_error: torch.Tensor,
        valid_mask: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        component_logs: dict[str, float] = {}
        total = sq_error.new_tensor(0.0)
        for name, component_slice in flame_component_layout(sq_error.shape[-1]):
            component_loss = self._masked_mean(sq_error[:, :, component_slice], valid_mask)
            total = total + getattr(self, f"w_{name}") * component_loss
            component_logs[f"{prefix}_{name}"] = float(component_loss.item())
        return total, component_logs

    def __call__(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        pred_target: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_mask = (~padding_mask).unsqueeze(-1).float().to(pred_velocity.device)

        flow_sq_error = (pred_velocity - target_velocity) ** 2
        flow_loss, flow_logs = self._weighted_component_loss(flow_sq_error, valid_mask, prefix="loss_flow")

        rec_sq_error = (pred_target - target) ** 2
        rec_loss, _ = self._weighted_component_loss(rec_sq_error, valid_mask, prefix="loss_rec_component")

        if pred_target.shape[1] > 1:
            valid_velocity = (~padding_mask[:, 1:] & ~padding_mask[:, :-1]).unsqueeze(-1).float().to(pred_target.device)
            pred_vel = pred_target[:, 1:] - pred_target[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            vel_loss = ((pred_vel - target_vel) ** 2 * valid_velocity).sum() / (valid_velocity.sum().clamp_min(1.0) * pred_target.shape[-1])
        else:
            vel_loss = pred_target.new_tensor(0.0)

        total = self.flow_weight * flow_loss + self.reconstruction_weight * rec_loss + self.velocity_weight * vel_loss
        return total, {
            "loss_total": float(total.item()),
            "loss_flow": float(flow_loss.item()),
            "loss_rec": float(rec_loss.item()),
            "loss_vel": float(vel_loss.item()),
            **flow_logs,
        }


def _resolve_target(batch: dict[str, torch.Tensor], target_dim: int) -> torch.Tensor:
    return batch[flame_target_key(target_dim)]


def train_motion_flow_matching(
    model: MotionFlowMatchingModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: FlowMatchingLoss,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
    epoch: int | None = None,
    num_epochs: int | None = None,
    log_interval: int = 1,
) -> dict[str, float]:
    """Run one flow-matching training epoch."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = _resolve_target(batch, model.target_dim).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_velocity, aux = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
                target=target,
            )
            loss, logs = criterion(
                pred_velocity=pred_velocity,
                target_velocity=aux["target_velocity"],
                pred_target=aux["pred_target"],
                target=target,
                padding_mask=padding_mask,
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        logs["mean_time"] = float(aux["times"].mean().item())
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
def validate_motion_flow_matching(
    model: MotionFlowMatchingModel,
    loader,
    criterion: FlowMatchingLoss,
    device: torch.device,
    use_amp: bool = False,
    epoch: int | None = None,
    num_epochs: int | None = None,
    eval_label: str = "val",
    log_interval: int = 1,
) -> dict[str, float]:
    """Validation loop using a deterministic midpoint interpolation target."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    eval_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        left_audio = batch["left_audio_feat"].to(device)
        left_video = batch["left_video_frames"].to(device)
        target = _resolve_target(batch, model.target_dim).to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        times = torch.full((target.shape[0],), 0.5, device=device, dtype=torch.float32)
        source = torch.zeros_like(target)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_velocity, aux = model(
                left_audio_feat=left_audio,
                left_video_frames=left_video,
                lengths=lengths,
                padding_mask=padding_mask,
                target=target,
                times=times,
                source=source,
            )
            _, logs = criterion(
                pred_velocity=pred_velocity,
                target_velocity=aux["target_velocity"],
                pred_target=aux["pred_target"],
                target=target,
                padding_mask=padding_mask,
            )
        logs["mean_time"] = float(aux["times"].mean().item())
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
def evaluate_motion_flow_matching_metrics(
    model: MotionFlowMatchingModel,
    loader,
    device: torch.device,
    use_amp: bool = False,
    reference_seq_ids: list[str] | None = None,
    manifest: dict[str, dict[str, str]] | None = None,
) -> dict[str, float]:
    """Run the shared benchmark metric stack on sampled flow-matching outputs."""

    class _SamplerWrapper(nn.Module):
        def __init__(self, inner: MotionFlowMatchingModel):
            super().__init__()
            self.inner = inner

        def forward(self, left_audio_feat, left_video_frames, lengths, padding_mask=None):
            del lengths
            return self.inner.sample(left_audio_feat, left_video_frames, padding_mask), None

    return evaluate_motion_metrics(
        _SamplerWrapper(model),
        loader,
        device=device,
        target_variant=flame_target_variant(model.target_dim),
        use_amp=use_amp,
        reference_seq_ids=reference_seq_ids,
        manifest=manifest,
    )