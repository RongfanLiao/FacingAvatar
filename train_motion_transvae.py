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
    evaluate_motion_metrics,
    MotionOnlyTransformerVAE,
    MotionVAELoss,
    save_checkpoint,
    train_motion_transvae,
    validate_motion_transvae,
)
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM
from config import DEVICE, NUM_WORKERS, VIDEO_CANVAS_SIZE, WAV2VEC_DIM
from manifest import load_documentary_manifest, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train motion-only TransVAE with raw video + wav2vec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--div_p", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--checkpoint_dir", default="checkpoints/motion_transvae_documentary")
    parser.add_argument("--split_path", default=default_benchmark_split_path())
    parser.add_argument("--train_val_same", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--target_variant", choices=["content", "motion58"], default="content")
    parser.add_argument("--video_canvas_size", type=int, default=VIDEO_CANVAS_SIZE)
    parser.add_argument("--documentary", action="store_true", help="Use documentary data manifest")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of val samples for evaluation")
    parser.add_argument("--predefined_splits_dir", type=str, default=None,
                        help="Path to directory with train.json/valid.json/test.json predefined splits")
    parser.add_argument("--val_interval", type=int, default=5, help="Validate every N epochs")
    return parser.parse_args()


def make_loader(
    seq_ids: list[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    target_variant: str,
    video_canvas_size: int,
    manifest: dict[str, dict[str, str]] | None = None,
) -> DataLoader:
    dataset = LookingFaceBenchmarkDataset(
        seq_ids=seq_ids,
        load_left_audio=False,
        load_wav2vec_audio=True,
        load_left_video_embedding=False,
        load_left_video_raw=True,
        video_canvas_size=video_canvas_size,
        load_flame_target=True,
        include_motion58_target=(target_variant == "motion58"),
        include_content_target=(target_variant == "content"),
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
        val_seqs = splits.get("test", splits.get("valid", []))
        print(f"Predefined splits: train={len(train_seqs)}, eval={len(val_seqs)}")
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
            target_variant=args.target_variant, video_canvas_size=args.video_canvas_size,
            manifest=manifest,
        )
    val_loader = make_loader(
        val_seqs, args.batch_size, args.num_workers, shuffle=False,
        target_variant=args.target_variant, video_canvas_size=args.video_canvas_size,
        manifest=manifest,
    )

    output_dim = FLAME_CONTENT_DIM if args.target_variant == "content" else FLAME_58_DIM

    model = MotionOnlyTransformerVAE(
        audio_dim=WAV2VEC_DIM,
        output_dim=output_dim,
        feature_dim=args.feature_dim,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
    ).to(device)
    criterion = MotionVAELoss(target_variant=args.target_variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if not args.eval_only:
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_motion_transvae(model, train_loader, optimizer, criterion, device=device, div_p=args.div_p, scaler=scaler)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_total={train_metrics['loss_total_with_div']:.5f} | "
                f"train_rec={train_metrics['loss_rec']:.5f} | "
                f"train_kld={train_metrics['loss_kld']:.5f}"
            )

            if epoch % args.val_interval == 0:
                val_metrics = validate_motion_transvae(model, val_loader, criterion, device=device, use_amp=use_amp)
                print(
                    f"  val_total={val_metrics['loss_total']:.5f} | "
                    f"val_rec={val_metrics['loss_rec']:.5f} | "
                    f"val_kld={val_metrics['loss_kld']:.5f}"
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

    metric_results = evaluate_motion_metrics(model, val_loader, device=device, target_variant=args.target_variant, use_amp=use_amp)
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
