"""Compute LPIPS and Fréchet video distance for rendered LookingFace videos.

This script pairs rendered prediction videos like:

    output/test_gen_vis/<seq_id>_right/<seq_id>_right.mp4

with the corresponding ground-truth `right_mp4` from the manifest and reports:

- `lpips`: mean frame-level LPIPS across all matched videos
- `fvd`: Fréchet distance between sets of video features extracted by a
  pretrained torchvision video backbone

The Fréchet distance here uses `r3d_18` features by default. That is not the
original I3D-based FVD implementation, but it serves the same role as a
distribution-level video similarity metric while staying self-contained in the
current environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu as decord_cpu
from scipy.linalg import sqrtm
from torchvision.models.video import R3D_18_Weights, r3d_18


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEVICE
from manifest import load_manifest


DEFAULT_RENDER_DIR = "output/test_gen_vis"
DEFAULT_OUTPUT_JSON = "output/test_gen_vis_metrics.json"


@dataclass(frozen=True)
class VideoPair:
    seq_id: str
    predicted_mp4: Path
    reference_mp4: Path


class VideoFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = R3D_18_Weights.DEFAULT
        backbone = r3d_18(weights=weights)
        backbone.fc = nn.Identity()
        self.model = backbone.eval()
        self.transforms = weights.transforms()

    def forward(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        processed = self.transforms(frames_tchw)
        return self.model(processed.unsqueeze(0)).squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rendered videos with LPIPS and Fréchet video distance")
    parser.add_argument("--render_dir", type=str, default=DEFAULT_RENDER_DIR, help="Directory containing per-sequence render outputs")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Path to save aggregate metrics JSON")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rendered videos to evaluate")
    parser.add_argument("--start_index", type=int, default=0, help="Optional starting index in the sorted rendered-video list")
    parser.add_argument("--device", type=str, default=DEVICE, help="Torch device for LPIPS and video features")
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex", help="Backbone used by LPIPS")
    parser.add_argument("--lpips_frames", type=int, default=16, help="Number of uniformly sampled frames per video for LPIPS")
    parser.add_argument("--lpips_size", type=int, default=224, help="Spatial size used for LPIPS frame comparisons")
    parser.add_argument("--fvd_frames", type=int, default=16, help="Number of uniformly sampled frames per video for Fréchet video features")
    parser.add_argument("--per_video_json", type=str, default=None, help="Optional path to save per-video metrics JSON")
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def sample_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    num_samples = max(1, min(num_samples, total_frames))
    return np.linspace(0, total_frames - 1, num_samples).round().astype(np.int64)


def load_frames(path: Path, indices: np.ndarray) -> np.ndarray:
    reader = VideoReader(str(path), ctx=decord_cpu(0))
    batch = reader.get_batch(indices.tolist()).asnumpy()
    return batch


def resize_frames_torch(frames: torch.Tensor, size: int) -> torch.Tensor:
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = torch.nn.functional.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
    return frames


def compute_lpips_for_pair(
    loss_fn: lpips.LPIPS,
    predicted_mp4: Path,
    reference_mp4: Path,
    num_frames: int,
    spatial_size: int,
    device: torch.device,
) -> float:
    pred_reader = VideoReader(str(predicted_mp4), ctx=decord_cpu(0))
    ref_reader = VideoReader(str(reference_mp4), ctx=decord_cpu(0))
    common_frames = min(len(pred_reader), len(ref_reader))
    if common_frames <= 0:
        raise ValueError(f"One of the videos is empty: {predicted_mp4} vs {reference_mp4}")

    indices = sample_indices(common_frames, num_frames)
    pred_frames = pred_reader.get_batch(indices.tolist()).asnumpy()
    ref_frames = ref_reader.get_batch(indices.tolist()).asnumpy()

    pred_tensor = resize_frames_torch(torch.from_numpy(pred_frames), spatial_size)
    ref_tensor = resize_frames_torch(torch.from_numpy(ref_frames), spatial_size)
    pred_tensor = pred_tensor.mul(2.0).sub(1.0).to(device)
    ref_tensor = ref_tensor.mul(2.0).sub(1.0).to(device)

    with torch.no_grad():
        scores = loss_fn(pred_tensor, ref_tensor)
    return float(scores.mean().item())


def compute_video_feature(
    extractor: VideoFeatureExtractor,
    video_path: Path,
    num_frames: int,
    device: torch.device,
) -> np.ndarray:
    reader = VideoReader(str(video_path), ctx=decord_cpu(0))
    indices = sample_indices(len(reader), num_frames)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(device=device)

    with torch.no_grad():
        feature = extractor(frames_tensor)
    return feature.detach().cpu().numpy().astype(np.float64)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    sigma_a = np.atleast_2d(np.cov(features_a, rowvar=False))
    sigma_b = np.atleast_2d(np.cov(features_b, rowvar=False))

    eps = 1e-6
    sigma_a = sigma_a + np.eye(sigma_a.shape[0], dtype=np.float64) * eps
    sigma_b = sigma_b + np.eye(sigma_b.shape[0], dtype=np.float64) * eps

    mean_diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = mean_diff.dot(mean_diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean)
    if not np.isfinite(value):
        raise ValueError("Fréchet distance became non-finite")
    return float(np.real(value))


def discover_pairs(render_dir: Path, manifest: dict[str, dict[str, str]]) -> list[VideoPair]:
    pairs: list[VideoPair] = []
    for seq_dir in sorted(path for path in render_dir.iterdir() if path.is_dir() and path.name.endswith("_right")):
        seq_id = seq_dir.name.removesuffix("_right")
        predicted_mp4 = seq_dir / f"{seq_dir.name}.mp4"
        if not predicted_mp4.exists():
            continue
        entry = manifest.get(seq_id)
        if entry is None:
            continue
        reference_mp4 = Path(entry.get("right_mp4") or "")
        if not reference_mp4.exists():
            continue
        pairs.append(VideoPair(seq_id=seq_id, predicted_mp4=predicted_mp4, reference_mp4=reference_mp4))
    return pairs


def ensure_parent(path: Path) -> None:
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    render_dir = resolve_repo_path(args.render_dir)
    output_json = resolve_repo_path(args.output_json)
    per_video_json = resolve_repo_path(args.per_video_json) if args.per_video_json else None

    manifest = load_manifest()
    pairs = discover_pairs(render_dir, manifest)
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.start_index:
        pairs = pairs[args.start_index:]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError(f"No rendered videos with matching manifest entries were found under {render_dir}")

    requested_device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    feature_extractor = VideoFeatureExtractor().to(device).eval()

    per_video: list[dict[str, object]] = []
    pred_features: list[np.ndarray] = []
    ref_features: list[np.ndarray] = []
    lpips_values: list[float] = []

    print(f"Evaluating {len(pairs)} rendered videos from {render_dir}")
    print(f"Using device: {device}")
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {pair.seq_id}")
        lpips_score = compute_lpips_for_pair(
            loss_fn=lpips_model,
            predicted_mp4=pair.predicted_mp4,
            reference_mp4=pair.reference_mp4,
            num_frames=args.lpips_frames,
            spatial_size=args.lpips_size,
            device=device,
        )
        pred_feature = compute_video_feature(feature_extractor, pair.predicted_mp4, args.fvd_frames, device)
        ref_feature = compute_video_feature(feature_extractor, pair.reference_mp4, args.fvd_frames, device)

        lpips_values.append(lpips_score)
        pred_features.append(pred_feature)
        ref_features.append(ref_feature)
        per_video.append(
            {
                "seq_id": pair.seq_id,
                "predicted_mp4": str(pair.predicted_mp4),
                "reference_mp4": str(pair.reference_mp4),
                "lpips": lpips_score,
            }
        )
        print(f"  lpips={lpips_score:.6f}")

    pred_matrix = np.stack(pred_features, axis=0)
    ref_matrix = np.stack(ref_features, axis=0)
    fvd_value = frechet_distance(pred_matrix, ref_matrix)
    lpips_mean = float(np.mean(lpips_values))
    lpips_std = float(np.std(lpips_values))

    aggregate = {
        "num_videos": len(pairs),
        "render_dir": str(render_dir),
        "device": str(device),
        "lpips": lpips_mean,
        "lpips_std": lpips_std,
        "lpips_backbone": args.lpips_net,
        "lpips_frames": int(args.lpips_frames),
        "lpips_size": int(args.lpips_size),
        "fvd": fvd_value,
        "fvd_backbone": "r3d_18",
        "fvd_frames": int(args.fvd_frames),
    }

    ensure_parent(output_json)
    output_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Saved aggregate metrics to {output_json}")
    print(json.dumps(aggregate, indent=2))

    if per_video_json is not None:
        ensure_parent(per_video_json)
        per_video_json.write_text(json.dumps(per_video, indent=2) + "\n", encoding="utf-8")
        print(f"Saved per-video metrics to {per_video_json}")


if __name__ == "__main__":
    main()