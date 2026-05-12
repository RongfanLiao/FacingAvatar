"""Train the LookingFace REGNN benchmark port."""

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
from benchmark.motion_transvae import (
    checkpoint_state_dict,
    extract_checkpoint_metric,
    load_checkpoint,
    resume_training_state,
    save_checkpoint,
)
from benchmark.regnn import (
    LookingFaceREGNN,
    REGNNLoss,
    evaluate_saved_regnn_metrics,
    save_regnn_predictions,
    train_regnn,
    validate_regnn,
)
from benchmark.targets import flame_target_variant
from config import DEVICE, NUM_WORKERS, VIDEO_CANVAS_SIZE
from manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LookingFace REGNN benchmark port")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_period", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--fused_dim", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--edge_dim", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--act_type", choices=["ELU", "ReLU", "GeLU", "None"], default="ELU")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise_threshold", type=float, default=0.0)
    parser.add_argument("--latent_weight", type=float, default=1.0)
    parser.add_argument("--neighbor_pattern", choices=["all", "nearest"], default="all")
    parser.add_argument("--no_mid_loss", action="store_true")
    parser.add_argument("--mid_weight", type=float, default=1.0)
    parser.add_argument("--logdet_weight", type=float, default=0.0)
    parser.add_argument("--reconstruction_weight", type=float, default=0.0)
    parser.add_argument("--vel_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=1, help="Print per-iteration progress every N batches")
    parser.add_argument("--checkpoint_dir", default="checkpoints/regnn_port")
    parser.add_argument("--split_path", default=default_benchmark_split_path())
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--train_val_same", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--video_canvas_size", type=int, default=VIDEO_CANVAS_SIZE)
    parser.add_argument("--predefined_splits_dir", type=str, default="data/LookingFace/dataset_splits",
                        help="Path to directory with train.json/valid.json/test.json predefined splits")
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
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    eval_label = "val"
    manifest = load_manifest()
    reference_seq_ids: list[str] = []

    if args.predefined_splits_dir:
        splits = load_predefined_splits(
            splits_dir=args.predefined_splits_dir,
            manifest=manifest,
            require_wav2vec_audio=True,
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
        train_seqs, val_seqs = build_benchmark_split(
            split_path=args.split_path,
            require_left_audio=False,
            require_left_video_embedding=False,
            require_wav2vec_audio=True,
            manifest=manifest,
        )
    reference_seq_ids = list(train_seqs)
    if args.max_sequences > 0:
        train_seqs = train_seqs[: args.max_sequences]
        val_seqs = val_seqs[: args.max_sequences]
    if args.train_val_same:
        val_seqs = train_seqs

    if not train_seqs and not args.eval_only:
        raise RuntimeError(
            "No training sequences available. For REGNN this usually means "
            "wav2vec audio features are missing or the raw video files cannot be loaded."
        )
    if not val_seqs:
        raise RuntimeError(
            f"No {eval_label} sequences available. Check the predefined split files and required inputs."
        )

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

    model = LookingFaceREGNN(
        target_dim=118,
        fused_dim=args.fused_dim,
        num_frames=args.num_frames,
        edge_dim=args.edge_dim,
        neighbors=args.neighbors,
        layers=args.layers,
        act_type=args.act_type,
        dropout=args.dropout,
        noise_threshold=(args.noise_threshold if args.noise_threshold > 0 else None),
    ).to(device)
    criterion = REGNNLoss(
        neighbor_pattern=args.neighbor_pattern,
        use_mid_loss=not args.no_mid_loss,
        latent_weight=args.latent_weight,
        mid_weight=args.mid_weight,
        reconstruction_weight=args.reconstruction_weight,
        vel_weight=args.vel_weight,
        logdet_weight=args.logdet_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            train_metrics = train_regnn(
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
                f"train_total={train_metrics['loss_total']:.5f} | "
                f"train_match={train_metrics['loss_match']:.5f} | "
                f"train_mid={train_metrics['loss_mid']:.5f}"
            )
            if epoch % args.val_period == 0:
                val_metrics = validate_regnn(
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
                    f"val_match={val_metrics['loss_match']:.5f} | "
                    f"val_mid={val_metrics['loss_mid']:.5f}"
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

    if args.max_sequences > 0:
        cache_scope = args.max_sequences
    else:
        cache_scope = "all"
    checkpoint_source = args.resume_checkpoint if args.resume_checkpoint else best_path
    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_source))[0]
    prediction_cache_name = f"{eval_label}_predictions_{checkpoint_tag}_{cache_scope}"
    prediction_cache_dir = os.path.join(args.checkpoint_dir, prediction_cache_name)
    prediction_summary = save_regnn_predictions(
        model,
        val_loader,
        device=device,
        output_dir=prediction_cache_dir,
        eval_label=eval_label,
        log_interval=args.log_interval,
    )
    metric_results = evaluate_saved_regnn_metrics(
        predictions_dir=prediction_cache_dir,
        seq_ids=val_seqs,
        manifest=manifest,
        model=model,
        reference_seq_ids=reference_seq_ids,
        eval_label=eval_label,
        log_interval=args.log_interval,
    )
    metric_results.update(prediction_summary)
    metric_results["target_variant"] = flame_target_variant(model.target_dim)
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