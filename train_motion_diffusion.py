"""Train the LookingFace motion_diffusion benchmark port."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from benchmark.lookingface import (
    LookingFaceBenchmarkDataset,
    build_benchmark_split,
    collate_benchmark_batch,
    default_benchmark_split_path,
)
from benchmark.motion_diffusion import (
    MotionDiffusionLoss,
    MotionDiffusionModel,
    evaluate_motion_diffusion_metrics,
    train_motion_diffusion,
    validate_motion_diffusion,
)
from benchmark.motion_transvae import save_checkpoint
from config import DEVICE, NUM_WORKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LookingFace motion_diffusion benchmark port")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_period", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_timesteps", type=int, default=1000)
    parser.add_argument("--inference_timesteps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--clip_sample", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--audio_drop_prob", type=float, default=0.2)
    parser.add_argument("--video_drop_prob", type=float, default=0.2)
    parser.add_argument("--latent_drop_prob", type=float, default=0.2)
    parser.add_argument("--timestep_spacing", choices=["leading", "linspace", "trailing", "full"], default="leading")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--checkpoint_dir", default="checkpoints/motion_diffusion_port")
    parser.add_argument("--split_path", default=default_benchmark_split_path())
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--train_val_same", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--target_variant", choices=["content", "motion58"], default="content")
    return parser.parse_args()


def make_loader(seq_ids: list[str], batch_size: int, num_workers: int, shuffle: bool, target_variant: str) -> DataLoader:
    dataset = LookingFaceBenchmarkDataset(
        seq_ids=seq_ids,
        load_left_audio=True,
        load_left_video_embedding=True,
        load_flame_target=True,
        include_motion58_target=(target_variant == "motion58"),
        include_content_target=(target_variant == "content"),
        require_right_mp4=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_benchmark_batch,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_seqs, val_seqs = build_benchmark_split(split_path=args.split_path)
    if args.max_sequences > 0:
        train_seqs = train_seqs[: args.max_sequences]
        val_seqs = val_seqs[: args.max_sequences]
    if args.train_val_same:
        val_seqs = train_seqs

    train_loader = make_loader(train_seqs, args.batch_size, args.num_workers, shuffle=True, target_variant=args.target_variant)
    val_loader = make_loader(val_seqs, args.batch_size, args.num_workers, shuffle=False, target_variant=args.target_variant)

    model = MotionDiffusionModel(
        target_variant=args.target_variant,
        feature_dim=args.feature_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        train_timesteps=args.train_timesteps,
        inference_timesteps=args.inference_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        clip_sample=args.clip_sample,
        guidance_scale=args.guidance_scale,
        audio_drop_prob=args.audio_drop_prob,
        video_drop_prob=args.video_drop_prob,
        latent_drop_prob=args.latent_drop_prob,
        timestep_spacing=args.timestep_spacing,
        ddim_eta=args.ddim_eta,
    ).to(device)
    criterion = MotionDiffusionLoss(target_variant=args.target_variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    if not args.eval_only:
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_motion_diffusion(
                model,
                train_loader,
                optimizer,
                criterion,
                device=device,
                grad_clip=args.grad_clip,
            )
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_total={train_metrics['loss_total_with_clip']:.5f} | "
                f"train_rec={train_metrics['loss_rec']:.5f} | "
                f"train_vel={train_metrics['loss_vel']:.5f}"
            )

            if epoch % args.val_period == 0:
                val_metrics = validate_motion_diffusion(model, val_loader, criterion, device=device)
                print(
                    f"  val_total={val_metrics['loss_total']:.5f} | "
                    f"val_rec={val_metrics['loss_rec']:.5f} | "
                    f"val_vel={val_metrics['loss_vel']:.5f}"
                )
                last_path = os.path.join(args.checkpoint_dir, "last.pt")
                save_checkpoint(last_path, model, optimizer, epoch, val_metrics)
                if val_metrics["loss_total"] < best_val:
                    best_val = val_metrics["loss_total"]
                    save_checkpoint(best_path, model, optimizer, epoch, val_metrics)

    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    metric_results = evaluate_motion_diffusion_metrics(model, val_loader, device=device, target_variant=args.target_variant)
    metric_results["target_variant"] = args.target_variant
    metric_path = os.path.join(args.checkpoint_dir, "metrics.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(metric_path, "w") as f:
        json.dump(metric_results, f, indent=2)

    print("Final validation metrics:")
    for key, value in metric_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}={value:.6f}")
        else:
            print(f"  {key}={value}")
    print(f"Metrics saved to: {metric_path}")


if __name__ == "__main__":
    main()