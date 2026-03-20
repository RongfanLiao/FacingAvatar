"""
Training script for Audio-Visual FLAME parameter prediction.

Usage:
    micromamba activate qwen_vl
    python train.py
"""

import os
import time

import torch
import torch.nn as nn

from config import (
    LR, WEIGHT_DECAY, BETAS, NUM_EPOCHS, WARMUP_STEPS,
    VEL_LOSS_WEIGHT, CKPT_DIR, DEVICE,
    FLAME_EXPR_DIM, FLAME_JAW_DIM, FLAME_ROT_DIM,
    FLAME_NECK_DIM, FLAME_EYES_DIM, FLAME_TRANSLATION_DIM,
    PARAM_LOSS_WEIGHTS, VEL_PARAM_LOSS_WEIGHTS,
)
from model import AudioVisualFLAMEModel
from dataset import build_dataloaders

os.makedirs(CKPT_DIR, exist_ok=True)

# ── FLAME param slices (order must match dataset.py FLAME_KEYS) ──────────────
_o1 = FLAME_EXPR_DIM
_o2 = _o1 + FLAME_JAW_DIM
_o3 = _o2 + FLAME_ROT_DIM
_o4 = _o3 + FLAME_NECK_DIM
_o5 = _o4 + FLAME_EYES_DIM
_o6 = _o5 + FLAME_TRANSLATION_DIM

PARAM_SLICES = {
    "expr":        slice(0, _o1),     # 0:100
    "jaw_pose":    slice(_o1, _o2),   # 100:103
    "rotation":    slice(_o2, _o3),   # 103:106
    "neck_pose":   slice(_o3, _o4),   # 106:109
    "eyes_pose":   slice(_o4, _o5),   # 109:115
    "translation": slice(_o5, _o6),   # 115:118
}


def compute_loss(
    preds: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute L1 parameter loss + velocity regularization.

    Args:
        preds: dict of (B, T, dim) predicted FLAME params
        target: (B, T, 118) ground-truth concatenated FLAME params
        mask: (B, T) padding mask — True for padded positions
        lengths: (B,) actual sequence lengths
    """
    l1 = nn.L1Loss(reduction="none")
    valid_mask = ~mask  # True for valid positions

    total_param_loss = 0.0
    total_vel_loss = 0.0
    log = {}

    for key, slc in PARAM_SLICES.items():
        pred = preds[key]                          # (B, T, dim)
        gt = target[:, :, slc]                     # (B, T, dim)

        # Parameter L1 loss (masked)
        param_loss = l1(pred, gt)                  # (B, T, dim)
        param_loss = param_loss * valid_mask.unsqueeze(-1)
        param_loss = param_loss.sum() / valid_mask.sum() / pred.size(-1)
        weighted_param_loss = PARAM_LOSS_WEIGHTS.get(key, 1.0) * param_loss
        total_param_loss = total_param_loss + weighted_param_loss
        log[f"l1_{key}"] = param_loss.item()
        log[f"w_l1_{key}"] = weighted_param_loss.item()

        # Velocity loss (temporal smoothness)
        pred_vel = pred[:, 1:] - pred[:, :-1]      # (B, T-1, dim)
        gt_vel = gt[:, 1:] - gt[:, :-1]
        vel_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        vel_loss = l1(pred_vel, gt_vel)
        vel_loss = vel_loss * vel_mask.unsqueeze(-1)
        if vel_mask.sum() > 0:
            vel_loss = vel_loss.sum() / vel_mask.sum() / pred.size(-1)
        else:
            vel_loss = vel_loss.sum() * 0
        weighted_vel_loss = VEL_PARAM_LOSS_WEIGHTS.get(key, 1.0) * vel_loss
        total_vel_loss = total_vel_loss + weighted_vel_loss
        log[f"vel_{key}"] = vel_loss.item()
        log[f"w_vel_{key}"] = weighted_vel_loss.item()

    total = total_param_loss + VEL_LOSS_WEIGHT * total_vel_loss
    log["loss_param"] = total_param_loss.item()
    log["loss_vel"] = total_vel_loss.item()
    log["loss_total"] = total.item()
    return total, log


def get_lr(step: int, warmup: int, base_lr: float, total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> dict[str, float]:
    """Run evaluation and return average metrics."""
    model.eval()
    totals = {}
    n_batches = 0
    for batch in loader:
        audio = batch["audio_feat"].to(device)
        video = batch["video_feat"].to(device)
        target = batch["flame_target"].to(device)
        mask = batch["padding_mask"].to(device)
        lengths = batch["lengths"].to(device)

        preds = model(audio, video, src_key_padding_mask=mask)
        _, log = compute_loss(preds, target, mask, lengths)

        for k, v in log.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train():
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders()

    print("Initializing model...")
    model = AudioVisualFLAMEModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )

    total_steps = NUM_EPOCHS * len(train_loader)
    best_val_loss = float("inf")
    global_step = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs ({total_steps} steps)...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_log = {}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            audio = batch["audio_feat"].to(DEVICE)
            video = batch["video_feat"].to(DEVICE)
            target = batch["flame_target"].to(DEVICE)
            mask = batch["padding_mask"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            # LR schedule
            lr = get_lr(global_step, WARMUP_STEPS, LR, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            preds = model(audio, video, src_key_padding_mask=mask)
            loss, log = compute_loss(preds, target, mask, lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in log.items():
                epoch_log[k] = epoch_log.get(k, 0.0) + v
            n_batches += 1
            global_step += 1

        # Average training metrics
        train_metrics = {k: v / max(n_batches, 1) for k, v in epoch_log.items()}
        dt = time.time() - t0

        # Validation
        if epoch % 10 == 0:
            val_metrics = evaluate(model, val_loader, DEVICE)

            print(
                f"Epoch {epoch:4d}/{NUM_EPOCHS} | "
                f"lr={lr:.2e} | "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"(param={train_metrics['loss_param']:.4f} vel={train_metrics['loss_vel']:.4f}) | "
                f"val_loss={val_metrics['loss_total']:.4f} | "
                f"{dt:.1f}s"
            )

            # Save best model
            if val_metrics["loss_total"] < best_val_loss:
                best_val_loss = val_metrics["loss_total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    os.path.join(CKPT_DIR, "best_model.pt"),
                )

        # Save periodic checkpoint
        if epoch % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss_total"],
                },
                os.path.join(CKPT_DIR, f"epoch_{epoch}.pt"),
            )

    # Save final model
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "val_loss": val_metrics["loss_total"],
        },
        os.path.join(CKPT_DIR, "final_model.pt"),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {CKPT_DIR}")


if __name__ == "__main__":
    train()
