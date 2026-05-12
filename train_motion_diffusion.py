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
    load_predefined_splits,
)
from benchmark.motion_diffusion import (
    MotionDiffusionLoss,
    MotionDiffusionModel,
    evaluate_saved_motion_diffusion_metrics,
    save_motion_diffusion_predictions,
    train_motion_diffusion,
    validate_motion_diffusion,
)
from benchmark.motion_transvae import (
    checkpoint_state_dict,
    extract_checkpoint_metric,
    load_checkpoint,
    resume_training_state,
    save_checkpoint,
)
from benchmark.targets import flame_target_variant
from config import DEVICE, NUM_WORKERS, VIDEO_CANVAS_SIZE
from manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LookingFace motion_diffusion benchmark port")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--predefined_splits_dir", type=str, default="data/LookingFace/dataset_splits",
                        help="Path to directory with train.json/valid.json/test.json predefined splits")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of val samples for evaluation")
    parser.add_argument("--train_val_same", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fast_eval", action="store_true",
                        help="Use a reduced diffusion step count for faster evaluation runs")
    parser.add_argument("--fast_eval_timesteps", type=int, default=10,
                        help="Reverse diffusion steps to use when --fast_eval is enabled")
    parser.add_argument("--video_canvas_size", type=int, default=VIDEO_CANVAS_SIZE)
    parser.add_argument("--log_interval", type=int, default=1, help="Print per-iteration progress every N batches")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Resume training from a saved checkpoint or raw state dict")
    return parser.parse_args()


def make_loader(
    seq_ids: list[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    video_canvas_size: int,
    manifest: dict[str, dict[str, str]] | None = None,
) -> DataLoader:
    dataset = LookingFaceBenchmarkDataset(
        seq_ids=seq_ids,
        load_left_wav2vec_audio=True,
        load_left_video_embedding=False,
        load_left_video_raw=True,
        video_canvas_size=video_canvas_size,
        load_flame_target=True,
        include_content_target=False,
        require_right_mp4=True,
        manifest=manifest,
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
    if args.fast_eval:
        if args.fast_eval_timesteps <= 0:
            raise ValueError("--fast_eval_timesteps must be positive")
        original_inference_timesteps = args.inference_timesteps
        args.inference_timesteps = min(args.inference_timesteps, args.fast_eval_timesteps)
        print(
            "Fast eval enabled: "
            f"inference_timesteps {original_inference_timesteps} -> {args.inference_timesteps}"
        )
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    eval_label = "val"
    manifest = load_manifest()
    reference_seq_ids: list[str] = []

    if args.predefined_splits_dir:
        splits = load_predefined_splits(
            splits_dir=args.predefined_splits_dir,
            manifest=manifest,
            require_wav2vec_audio=not args.eval_only,
        )
        train_seqs = splits.get("train", [])
        if "test" in splits:
            eval_label = "test"
            val_seqs = splits["test"]
        else:
            eval_label = "valid"
            val_seqs = splits.get("valid", [])
        print(f"Predefined splits: train={len(train_seqs)}, {eval_label}={len(val_seqs)}")
    else:
        train_seqs, val_seqs = build_benchmark_split(split_path=args.split_path, manifest=manifest)

    reference_seq_ids = list(train_seqs)

    if args.max_sequences > 0:
        train_seqs = train_seqs[: args.max_sequences]
        val_seqs = val_seqs[: args.max_sequences]
    if args.train_val_same:
        val_seqs = train_seqs
    if args.max_eval_samples is not None:
        val_seqs = val_seqs[:args.max_eval_samples]

    if not train_seqs and not args.eval_only:
        raise RuntimeError(
            "No training sequences available. For motion_diffusion this usually means "
            "wav2vec audio features are missing or the raw video files cannot be loaded."
        )
    if not val_seqs:
        raise RuntimeError(
            f"No {eval_label} sequences available. Check the predefined split files and required embeddings."
        )

    train_loader = None
    if not args.eval_only:
        train_loader = make_loader(
            train_seqs,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            video_canvas_size=args.video_canvas_size,
            manifest=manifest,
        )
    val_loader = make_loader(
        val_seqs,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        video_canvas_size=args.video_canvas_size,
        manifest=manifest,
    )

    model = MotionDiffusionModel(
        target_dim=118,
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
    criterion = MotionDiffusionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    start_epoch = 0
    resume_metric: float | None = None

    if args.resume_checkpoint:
        resume_state = resume_training_state(
            args.resume_checkpoint,
            model,
            device=device,
            optimizer=optimizer,
        )
        start_epoch = int(resume_state["epoch"])
        resume_metric = resume_state["primary_metric"]  # type: ignore[assignment]
        print(
            f"Resumed from {args.resume_checkpoint}: epoch={start_epoch}, "
            f"optimizer_restored={resume_state['restored_optimizer']}"
        )

    if not args.eval_only:
        best_val = float("inf")
        if os.path.exists(best_path):
            best_metric = extract_checkpoint_metric(load_checkpoint(best_path, device))
            if best_metric is not None:
                best_val = best_metric
        elif resume_metric is not None:
            best_val = resume_metric

        if start_epoch >= args.epochs:
            print(
                f"Resume checkpoint is already at epoch {start_epoch}, "
                f"so no additional training will run for --epochs {args.epochs}."
            )

        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_metrics = train_motion_diffusion(
                model,
                train_loader,
                optimizer,
                criterion,
                device=device,
                grad_clip=args.grad_clip,
                epoch=epoch,
                num_epochs=args.epochs,
                log_interval=args.log_interval,
            )
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_total={train_metrics['loss_total_with_clip']:.5f} | "
                f"train_rec={train_metrics['loss_rec']:.5f} | "
                f"train_vel={train_metrics['loss_vel']:.5f}"
            )

            if epoch % args.val_interval == 0:
                val_metrics = validate_motion_diffusion(
                    model,
                    val_loader,
                    criterion,
                    device=device,
                    epoch=epoch,
                    num_epochs=args.epochs,
                    eval_label=eval_label,
                    log_interval=args.log_interval,
                )
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

    if args.eval_only and args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, device)
        model.load_state_dict(checkpoint_state_dict(checkpoint))
    elif os.path.exists(best_path):
        checkpoint = load_checkpoint(best_path, device)
        model.load_state_dict(checkpoint_state_dict(checkpoint))

    target_variant = flame_target_variant(model.target_dim)
    if args.max_eval_samples is not None:
        cache_scope = args.max_eval_samples
    elif args.max_sequences > 0:
        cache_scope = args.max_sequences
    else:
        cache_scope = "all"
    checkpoint_source = args.resume_checkpoint if args.resume_checkpoint else best_path
    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_source))[0]
    prediction_cache_name = f"{eval_label}_predictions_{checkpoint_tag}_{cache_scope}"
    prediction_cache_dir = os.path.join(args.checkpoint_dir, prediction_cache_name)
    prediction_summary = save_motion_diffusion_predictions(
        model,
        val_loader,
        device=device,
        output_dir=prediction_cache_dir,
        eval_label=eval_label,
        log_interval=args.log_interval,
    )
    metric_results = evaluate_saved_motion_diffusion_metrics(
        predictions_dir=prediction_cache_dir,
        seq_ids=val_seqs,
        manifest=manifest,
        model=model,
        reference_seq_ids=reference_seq_ids,
        eval_label=eval_label,
        log_interval=args.log_interval,
    )
    metric_results.update(prediction_summary)
    metric_results["target_variant"] = target_variant
    metric_results["evaluation_split"] = eval_label
    metric_results.update({
        f"{eval_label}_{key}": value
        for key, value in list(metric_results.items())
        if key not in {"target_variant", "evaluation_split"}
    })
    metric_path = os.path.join(args.checkpoint_dir, "metrics.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(metric_path, "w") as f:
        json.dump(metric_results, f, indent=2)

    print(f"Final {eval_label} metrics:")
    for key, value in metric_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}={value:.6f}")
        else:
            print(f"  {key}={value}")
    print(f"Metrics saved to: {metric_path}")


if __name__ == "__main__":
    main()