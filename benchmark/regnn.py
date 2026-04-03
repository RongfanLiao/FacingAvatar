"""LookingFace-compatible REGNN port with graph-based content prediction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.motion_transvae import SpeakerContextEncoder, evaluate_motion_metrics
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM


class MultiNodeMlp(nn.Module):
    """Baseline REGNN node-wise MLP adapted for fixed clip windows."""

    def __init__(self, in_dim: int, out_dim: int, n_nodes: int):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(n_nodes, in_dim, in_dim))
        self.bias1 = nn.Parameter(torch.zeros(n_nodes, 1, in_dim))
        self.weight2 = nn.Parameter(torch.empty(n_nodes, in_dim, out_dim))
        self.bias2 = nn.Parameter(torch.zeros(n_nodes, 1, out_dim))
        self.act = nn.GELU()
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def spectral_norm_(self) -> None:
        with torch.no_grad():
            norm1 = torch.linalg.norm(self.weight1, dim=(1, 2), ord=2, keepdim=True).clamp_min(1e-6)
            norm2 = torch.linalg.norm(self.weight2, dim=(1, 2), ord=2, keepdim=True).clamp_min(1e-6)
            self.weight1.data = self.weight1 / norm1
            self.weight2.data = self.weight2 / norm2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.spectral_norm_()
        outputs = torch.matmul(inputs.unsqueeze(2), self.weight1) + self.bias1
        outputs = self.act(outputs.squeeze(2)).unsqueeze(2)
        outputs = torch.matmul(outputs, self.weight2) + self.bias2
        return outputs.squeeze(2)


class EdgeLayer(nn.Module):
    """Top-k cognitive graph edge builder from the baseline REGNN design."""

    def __init__(self, dim: int, n_channels: int, neighbors: int, bias: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.neighbors = neighbors
        self.scale = dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2 * n_channels, bias=bias)

    def norm_edge(self, edge: torch.Tensor) -> torch.Tensor:
        norm_row = edge / (edge.sum(dim=-1, keepdim=True) + 1e-6)
        norm_col = norm_row / (norm_row.sum(dim=-2, keepdim=True) + 1e-6)
        return torch.matmul(norm_row, norm_col.transpose(-1, -2))

    def clip(self, edge: torch.Tensor) -> torch.Tensor:
        sum_edge = edge.detach().sum(dim=1)
        _, top_idx = torch.topk(sum_edge, k=min(self.neighbors, sum_edge.shape[-1]), dim=-1)
        masks = []
        for batch_idx in range(top_idx.shape[0]):
            mask = torch.eye(sum_edge.shape[-1], device=edge.device)
            for row_idx, neighbors in enumerate(top_idx[batch_idx]):
                mask[row_idx, neighbors] = 1.0
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).unsqueeze(1)
        return self.norm_edge(mask_tensor * edge)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = x.shape
        qk = self.qk(x).reshape(batch_size, num_nodes, 2, self.n_channels, feature_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.clip(attn)


class GraphAttention(nn.Module):
    """Lipschitz-friendly attention message passing over the learned edge graph."""

    def __init__(self, num_features: int, edge_channel: int, act_type: str):
        super().__init__()
        self.num_features = num_features
        self.edge_channel = edge_channel
        self.qk = nn.Linear(num_features, num_features * 2, bias=True)
        self.scale = num_features ** -0.5
        self.w = nn.Parameter(torch.ones(1, edge_channel))
        self.act_layer = {
            "ReLU": nn.ReLU(),
            "ELU": nn.ELU(),
            "GeLU": nn.GELU(),
            "None": nn.Identity(),
        }[act_type]
        self.lipnorm = 1.0

    def norm_edge(self, edge: torch.Tensor) -> torch.Tensor:
        return edge / (torch.sum(edge, dim=-1, keepdim=True) + 1e-6)

    def get_norm(self, weight: torch.Tensor) -> None:
        with torch.no_grad():
            weight_q = weight[: self.num_features]
            weight_k = weight[self.num_features :]
            dot_matrix = weight_q @ weight_k.T * self.scale
            self.lipnorm = float(torch.linalg.norm(torch.eye(self.num_features, device=weight.device) + 2 * dot_matrix, ord=2) + 5.0)

    def forward(self, x: torch.Tensor, edge: torch.Tensor, cal_norm: bool = True) -> torch.Tensor:
        x = torch.sigmoid(self.act_layer(x))
        if cal_norm:
            self.get_norm(self.qk.weight.data)

        batch_size, num_nodes, dim = x.shape
        qk = self.qk(x).reshape(batch_size, num_nodes, 2, dim).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        new_edge = self.norm_edge(attn.unsqueeze(1) * edge)
        new_edge = torch.einsum("bpnd,ip->bind", new_edge, self.w / self.w.sum()).squeeze(1)
        return torch.matmul(new_edge, x) / self.lipnorm


class GraphLayer(nn.Module):
    """Residual invertible graph block."""

    def __init__(self, num_features: int, edge_channel: int, act_type: str):
        super().__init__()
        self.attn = GraphAttention(num_features=num_features, edge_channel=edge_channel, act_type=act_type)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fx = self.attn(x, edge, cal_norm=True)
        return x + fx, x.new_tensor(0.0)

    def inverse(self, y: torch.Tensor, edge: torch.Tensor, cal_norm: bool) -> torch.Tensor:
        x = y
        for idx in range(5):
            fx = self.attn(x, edge, cal_norm=(idx == 0 and cal_norm))
            x = y - fx
        return x


class LipschitzGraph(nn.Module):
    """Motor processor from REGNN adapted for LookingFace targets."""

    def __init__(self, edge_channel: int, num_features: int, n_layers: int, act_type: str):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphLayer(num_features=num_features, edge_channel=edge_channel, act_type=act_type)
            for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = inputs
        total_logdet = x.new_tensor(0.0)
        for layer in self.layers:
            x, logdet = layer(x, edge)
            total_logdet = total_logdet + logdet
        return x, total_logdet

    def inverse(self, inputs: torch.Tensor, edge: torch.Tensor, cal_norm: bool = True) -> torch.Tensor:
        x = inputs
        for layer in reversed(self.layers):
            x = layer.inverse(x, edge=edge, cal_norm=cal_norm)
        return x


class REGNNCognitiveProcessor(nn.Module):
    """Convert fused temporal features into node-wise speaker graph features and edges."""

    def __init__(self, input_dim: int, output_nodes: int, num_frames: int, edge_dim: int, neighbors: int):
        super().__init__()
        self.convert_layer = MultiNodeMlp(in_dim=input_dim, out_dim=output_nodes, n_nodes=num_frames)
        self.edge_layer = EdgeLayer(dim=num_frames, n_channels=edge_dim, neighbors=neighbors)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        converted = self.convert_layer(inputs)
        node_features = converted.transpose(1, 2)
        edge = self.edge_layer(node_features)
        return node_features, edge


class LookingFacePercepProcessor(nn.Module):
    """Adapted perceptual fusion for LookingFace left audio plus static left video."""

    def __init__(self, audio_dim: int, video_dim: int, fused_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = SpeakerContextEncoder(
            audio_dim=audio_dim,
            video_dim=video_dim,
            feature_dim=fused_dim,
            n_heads=4,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, left_audio_feat: torch.Tensor, left_video_feat: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        return self.encoder(left_audio_feat, left_video_feat, padding_mask=padding_mask)


class LookingFaceREGNN(nn.Module):
    """Graph-based REGNN benchmark port for LookingFace."""

    def __init__(
        self,
        audio_dim: int = 1280,
        video_dim: int = 3584,
        fused_dim: int = 64,
        target_variant: str = "content",
        num_frames: int = 50,
        edge_dim: int = 8,
        neighbors: int = 6,
        layers: int = 2,
        act_type: str = "ELU",
        dropout: float = 0.1,
        noise_threshold: float | None = None,
    ):
        super().__init__()
        if target_variant == "content":
            self.target_dim = FLAME_CONTENT_DIM
        elif target_variant == "motion58":
            self.target_dim = FLAME_58_DIM
        else:
            raise ValueError(f"Unsupported target_variant: {target_variant}")
        self.target_variant = target_variant
        self.num_frames = num_frames
        self.noise_threshold = noise_threshold
        self.perceptual_processor = LookingFacePercepProcessor(audio_dim=audio_dim, video_dim=video_dim, fused_dim=fused_dim, dropout=dropout)
        self.cognitive_processor = REGNNCognitiveProcessor(
            input_dim=fused_dim,
            output_nodes=self.target_dim,
            num_frames=num_frames,
            edge_dim=edge_dim,
            neighbors=neighbors,
        )
        self.motor_processor = LipschitzGraph(
            edge_channel=edge_dim,
            num_features=num_frames,
            n_layers=layers,
            act_type=act_type,
        )

    def forward_features(
        self,
        left_audio_clip: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.perceptual_processor(left_audio_clip, left_video_feat, padding_mask=padding_mask)
        return self.cognitive_processor(fused)

    def sample(self, speaker_feature: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        noise = torch.randn_like(speaker_feature)
        if threshold is None:
            threshold = self.noise_threshold
        if threshold is None:
            return speaker_feature + noise
        max_abs = torch.max(torch.abs(noise)).clamp_min(1e-6)
        scaled_noise = noise * (math.sqrt(threshold) / max_abs)
        return speaker_feature + scaled_noise

    def forward(
        self,
        left_audio_clip: torch.Tensor,
        left_video_feat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_clip: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        speaker_feature, edge = self.forward_features(left_audio_clip, left_video_feat, padding_mask=padding_mask)
        outputs = {
            "speaker_feature": speaker_feature,
            "edge": edge,
        }
        decoded = self.motor_processor.inverse(speaker_feature, edge=edge, cal_norm=True).transpose(1, 2)
        outputs["prediction"] = decoded
        if target_clip is not None:
            target_nodes = target_clip.transpose(1, 2)
            listener_feature, logdets = self.motor_processor(target_nodes, edge=edge)
            outputs["listener_feature"] = listener_feature
            outputs["logdets"] = logdets
        return outputs

    @torch.no_grad()
    def predict_sequence(
        self,
        left_audio_feat: torch.Tensor,
        left_video_feat: torch.Tensor,
        lengths: torch.Tensor,
        target_variant: str,
    ) -> torch.Tensor:
        del target_variant
        predictions = []
        for batch_idx, length in enumerate(lengths.tolist()):
            seq_preds = []
            audio_seq = left_audio_feat[batch_idx, :length]
            video_feat = left_video_feat[batch_idx : batch_idx + 1]
            for start in range(0, length, self.num_frames):
                end = min(start + self.num_frames, length)
                clip = audio_seq[start:end]
                valid_len = clip.shape[0]
                if valid_len < self.num_frames:
                    pad = torch.zeros(self.num_frames - valid_len, clip.shape[-1], device=clip.device, dtype=clip.dtype)
                    clip = torch.cat([clip, pad], dim=0)
                clip = clip.unsqueeze(0)
                padding_mask = torch.arange(self.num_frames, device=clip.device).unsqueeze(0) >= valid_len
                output = self.forward(clip, video_feat, padding_mask=padding_mask, target_clip=None)["prediction"]
                seq_preds.append(output[0, :valid_len])
            predictions.append(torch.cat(seq_preds, dim=0))

        max_len = max(seq.shape[0] for seq in predictions)
        padded = []
        for seq in predictions:
            if seq.shape[0] < max_len:
                pad = torch.zeros(max_len - seq.shape[0], seq.shape[1], device=seq.device, dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)
        return torch.stack(padded, dim=0)


def _resolve_target(batch: dict[str, torch.Tensor], target_variant: str) -> torch.Tensor:
    return batch["flame_target_58"] if target_variant == "motion58" else batch["flame_target_content"]


def build_regnn_clips(
    batch: dict[str, torch.Tensor],
    target_variant: str,
    num_frames: int,
    random_start: bool,
) -> dict[str, torch.Tensor]:
    """Extract fixed-size clips from padded full sequences for REGNN training."""
    left_audio = batch["left_audio_feat"]
    left_video = batch["left_video_feat"]
    target = _resolve_target(batch, target_variant)
    lengths = batch["lengths"]

    audio_clips = []
    target_clips = []
    clip_lengths = []
    for batch_idx, seq_len in enumerate(lengths.tolist()):
        if seq_len <= num_frames:
            start = 0
        elif random_start:
            start = random.randint(0, seq_len - num_frames)
        else:
            start = 0
        end = min(start + num_frames, seq_len)
        valid_len = end - start

        audio_clip = left_audio[batch_idx, start:end]
        target_clip = target[batch_idx, start:end]
        if valid_len < num_frames:
            audio_pad = torch.zeros(num_frames - valid_len, audio_clip.shape[-1], dtype=audio_clip.dtype)
            target_pad = torch.zeros(num_frames - valid_len, target_clip.shape[-1], dtype=target_clip.dtype)
            audio_clip = torch.cat([audio_clip, audio_pad], dim=0)
            target_clip = torch.cat([target_clip, target_pad], dim=0)

        audio_clips.append(audio_clip)
        target_clips.append(target_clip)
        clip_lengths.append(valid_len)

    clip_lengths_tensor = torch.tensor(clip_lengths, dtype=torch.long)
    padding_mask = torch.arange(num_frames).unsqueeze(0) >= clip_lengths_tensor.unsqueeze(1)
    return {
        "left_audio_clip": torch.stack(audio_clips, dim=0),
        "left_video_feat": left_video,
        "target_clip": torch.stack(target_clips, dim=0),
        "clip_lengths": clip_lengths_tensor,
        "padding_mask": padding_mask,
    }


@dataclass
class REGNNLoss:
    """Baseline-shaped REGNN objective with latent matching as the primary loss."""

    target_variant: str = "content"
    neighbor_pattern: str = "all"
    threshold: float | None = None
    use_mid_loss: bool = True
    latent_weight: float = 1.0
    mid_weight: float = 1.0
    logdet_weight: float = 0.0
    reconstruction_weight: float = 0.0
    vel_weight: float = 0.0
    w_exp: float = 2.0
    w_jaw: float = 2.0
    w_neck: float = 2.0
    w_eyes: float = 2.0
    w_rot: float = 4.0
    w_tran: float = 4.0

    def _all_thre_mse_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths_list = lengths.tolist()
        preds = torch.repeat_interleave(preds, dim=0, repeats=torch.as_tensor(lengths_list, device=preds.device))
        all_mse = F.mse_loss(preds, targets, reduction="none").mean(dim=(1, 2))

        if self.neighbor_pattern == "nearest":
            seqs = torch.split(all_mse, lengths_list)
            mse = torch.stack([torch.min(seq) for seq in seqs])
            if self.threshold is not None:
                mse = torch.where(mse > self.threshold, mse, torch.tensor([0.0], device=mse.device, dtype=mse.dtype))
            return torch.mean(mse)

        seqs = torch.split(all_mse, lengths_list)
        mse = torch.stack([torch.min(seq) for seq in seqs])
        if self.threshold is not None:
            mse = torch.where(mse > self.threshold, mse, torch.tensor([0.0], device=mse.device, dtype=mse.dtype))
        return torch.mean(mse)

    def _mid_loss(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_list = lengths.tolist()
        seqs = torch.split(inputs, lengths_list)
        means = torch.stack([torch.mean(seq, dim=0) for seq in seqs])
        means = torch.repeat_interleave(means, dim=0, repeats=torch.as_tensor(lengths_list, device=means.device))
        return F.mse_loss(means, inputs)

    def _masked_reconstruction_loss(self, prediction: torch.Tensor, target: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        valid_mask = (~padding_mask).unsqueeze(-1).float().to(prediction.device)
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
        else:
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

        return rec_loss, component_logs

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        padding_mask: torch.Tensor,
        candidate_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = outputs["prediction"]
        speaker_feature = outputs["speaker_feature"]
        listener_feature = outputs.get("listener_feature")
        logdets = outputs.get("logdets")

        if listener_feature is None:
            raise ValueError("listener_feature is required for REGNN training loss")

        latent_loss = self._all_thre_mse_loss(speaker_feature, listener_feature, candidate_lengths)
        mid_loss = self._mid_loss(listener_feature, candidate_lengths) if self.use_mid_loss else prediction.new_tensor(0.0)

        rec_loss, component_logs = self._masked_reconstruction_loss(prediction, target, padding_mask)

        if prediction.shape[1] > 1:
            valid_velocity = (~padding_mask[:, 1:] & ~padding_mask[:, :-1]).unsqueeze(-1).float().to(prediction.device)
            pred_vel = prediction[:, 1:] - prediction[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            vel_loss = ((pred_vel - target_vel) ** 2 * valid_velocity).sum() / (valid_velocity.sum().clamp_min(1.0) * prediction.shape[-1])
        else:
            vel_loss = prediction.new_tensor(0.0)

        logdet_loss = prediction.new_tensor(0.0)
        if logdets is not None and not isinstance(logdets, float):
            logdet_loss = torch.as_tensor(logdets, device=prediction.device, dtype=prediction.dtype)

        total = (
            self.latent_weight * latent_loss
            + self.mid_weight * mid_loss
            + self.logdet_weight * logdet_loss
            + self.reconstruction_weight * rec_loss
            + self.vel_weight * vel_loss
        )
        return total, {
            "loss_total": float(total.item()),
            "loss_match": float(latent_loss.item()),
            "loss_mid": float(mid_loss.item()),
            "loss_logdet": float(logdet_loss.item()),
            "loss_rec": float(rec_loss.item()),
            "loss_vel": float(vel_loss.item()),
            **component_logs,
        }


def train_regnn(
    model: LookingFaceREGNN,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: REGNNLoss,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Run one REGNN training epoch on random fixed windows."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        clip_batch = build_regnn_clips(batch, criterion.target_variant, num_frames=model.num_frames, random_start=True)
        left_audio = clip_batch["left_audio_clip"].to(device)
        left_video = clip_batch["left_video_feat"].to(device)
        target = clip_batch["target_clip"].to(device)
        padding_mask = clip_batch["padding_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(left_audio, left_video, padding_mask=padding_mask, target_clip=target)
        candidate_lengths = torch.ones(target.shape[0], dtype=torch.long, device=device)
        loss, logs = criterion(outputs, target, padding_mask, candidate_lengths)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_regnn(
    model: LookingFaceREGNN,
    loader,
    criterion: REGNNLoss,
    device: torch.device,
) -> dict[str, float]:
    """Validate REGNN on deterministic first-window clips."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        clip_batch = build_regnn_clips(batch, criterion.target_variant, num_frames=model.num_frames, random_start=False)
        left_audio = clip_batch["left_audio_clip"].to(device)
        left_video = clip_batch["left_video_feat"].to(device)
        target = clip_batch["target_clip"].to(device)
        padding_mask = clip_batch["padding_mask"].to(device)

        outputs = model(left_audio, left_video, padding_mask=padding_mask, target_clip=target)
        candidate_lengths = torch.ones(target.shape[0], dtype=torch.long, device=device)
        _, logs = criterion(outputs, target, padding_mask, candidate_lengths)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_regnn_metrics(
    model: LookingFaceREGNN,
    loader,
    device: torch.device,
    target_variant: str = "content",
) -> dict[str, float]:
    """Run shared motion metrics on full-sequence REGNN predictions."""

    class _REGNNWrapper(nn.Module):
        def __init__(self, inner: LookingFaceREGNN, variant: str):
            super().__init__()
            self.inner = inner
            self.variant = variant

        def forward(self, left_audio_feat, left_video_feat, lengths, padding_mask=None):
            del padding_mask
            prediction = self.inner.predict_sequence(left_audio_feat, left_video_feat, lengths, target_variant=self.variant)
            return prediction, None

    return evaluate_motion_metrics(_REGNNWrapper(model, target_variant), loader, device=device, target_variant=target_variant)