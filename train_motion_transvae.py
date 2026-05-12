"""Train the motion-only TransVAE with raw video + wav2vec audio on documentary data."""

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
    evaluate_saved_motion_metrics,
    extract_checkpoint_metric,
    load_checkpoint,
    MotionTransformerVAE,
    MotionVAELoss,
    resume_training_state,
    save_motion_predictions,
    save_checkpoint,
    train_motion_transvae,
    validate_motion_transvae,
)
from benchmark.targets import FLAME_118_DIM, flame_target_variant
from config import DEVICE, NUM_WORKERS, VIDEO_CANVAS_SIZE, WAV2VEC_DIM
from manifest import load_documentary_manifest, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train motion-only TransVAE with raw video + wav2vec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--div_p", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--checkpoint_dir", default="checkpoints/motion_transvae")
    parser.add_argument("--split_path", default=default_benchmark_split_path())
    parser.add_argument("--train_val_same", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--video_canvas_size", type=int, default=VIDEO_CANVAS_SIZE)
    parser.add_argument("--documentary", action="store_true", help="Use documentary data manifest")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of val samples for evaluation")
    parser.add_argument("--predefined_splits_dir", type=str, default="data/LookingFace/dataset_splits",
                        help="Path to directory with train.json/valid.json/test.json predefined splits")
    parser.add_argument("--val_interval", type=int, default=4, help="Validate every N epochs")
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
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    eval_label = "val"
    last_epoch = 0
    last_checkpoint_metrics: dict[str, float] = {}
    validation_ran = False

    if args.documentary:
        manifest = load_documentary_manifest()
    else:
        manifest = load_manifest()

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
    if args.train_val_same:
        val_seqs = train_seqs
    if args.max_eval_samples is not None:
        val_seqs = val_seqs[:args.max_eval_samples]

    train_loader = None
    if not args.eval_only:
        train_loader = make_loader(
            train_seqs, args.batch_size, args.num_workers, shuffle=True,
            video_canvas_size=args.video_canvas_size,
            manifest=manifest,
        )
    val_loader = make_loader(
        val_seqs, args.batch_size, args.num_workers, shuffle=False,
        video_canvas_size=args.video_canvas_size,
        manifest=manifest,
    )

    output_dim = FLAME_118_DIM

    model = MotionTransformerVAE(
        audio_dim=WAV2VEC_DIM,
        output_dim=output_dim,
        feature_dim=args.feature_dim,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
    ).to(device)
    criterion = MotionVAELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    start_epoch = 0
    resume_metric: float | None = None

    if args.resume_checkpoint:
        resume_state = resume_training_state(
            args.resume_checkpoint,
            model,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        start_epoch = int(resume_state["epoch"])
        resume_metric = resume_state["primary_metric"]  # type: ignore[assignment]
        print(
            f"Resumed from {args.resume_checkpoint}: epoch={start_epoch}, "
            f"optimizer_restored={resume_state['restored_optimizer']}, "
            f"scaler_restored={resume_state['restored_scaler']}"
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
            last_epoch = epoch
            train_metrics = train_motion_transvae(
                model,
                train_loader,
                optimizer,
                criterion,
                device=device,
                div_p=args.div_p,
                scaler=scaler,
                epoch=epoch,
                num_epochs=args.epochs,
                log_interval=args.log_interval,
            )
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_total={train_metrics['loss_total_with_div']:.5f} | "
                f"train_rec={train_metrics['loss_rec']:.5f} | "
                f"train_kld={train_metrics['loss_kld']:.5f}"
            )

            if epoch % args.val_interval == 0:
                validation_ran = True
                val_metrics = validate_motion_transvae(
                    model,
                    val_loader,
                    criterion,
                    device=device,
                    use_amp=use_amp,
                    epoch=epoch,
                    num_epochs=args.epochs,
                    eval_label=eval_label,
                    log_interval=args.log_interval,
                )
                last_checkpoint_metrics = val_metrics
                print(
                    f"  {eval_label}_total={val_metrics['loss_total']:.5f} | "
                    f"{eval_label}_rec={val_metrics['loss_rec']:.5f} | "
                    f"{eval_label}_kld={val_metrics['loss_kld']:.5f}"
                )
                last_path = os.path.join(args.checkpoint_dir, "last.pt")
                save_checkpoint(last_path, model, optimizer, epoch, val_metrics)
                if val_metrics["loss_total"] < best_val:
                    best_val = val_metrics["loss_total"]
                    save_checkpoint(best_path, model, optimizer, epoch, val_metrics)

        final_path = os.path.join(args.checkpoint_dir, "final.pt")
        save_checkpoint(final_path, model, optimizer, last_epoch, last_checkpoint_metrics)
        print(f"Final checkpoint saved to: {final_path}")
        if not validation_ran:
            print(
                "Validation did not run during training, so best.pt/last.pt were not written. "
                "Use --val_interval <= --epochs if you want validation checkpoints."
            )

    if args.eval_only and args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, device)
        model.load_state_dict(checkpoint_state_dict(checkpoint))
    elif os.path.exists(best_path):
        checkpoint = load_checkpoint(best_path, device)
        model.load_state_dict(checkpoint_state_dict(checkpoint))

    target_variant = flame_target_variant(output_dim)
    checkpoint_source = args.resume_checkpoint if args.resume_checkpoint else best_path
    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_source))[0]
    prediction_cache_name = (
        f"{eval_label}_predictions_{checkpoint_tag}_{args.max_eval_samples if args.max_eval_samples is not None else 'all'}"
    )
    prediction_cache_dir = os.path.join(args.checkpoint_dir, prediction_cache_name)
    prediction_summary = save_motion_predictions(
        model,
        val_loader,
        device=device,
        use_amp=use_amp,
        output_dir=prediction_cache_dir,
        eval_label=eval_label,
        log_interval=args.log_interval,
    )
    metric_results = evaluate_saved_motion_metrics(
        predictions_dir=prediction_cache_dir,
        seq_ids=val_seqs,
        manifest=manifest,
        target_variant=target_variant,
        reference_seq_ids=train_seqs,
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
