"""Draft reaction type annotation for LookingFace paired videos using a local VLM.

This pilot runner pairs stimulus/reaction videos from the LookingFace manifest,
formats the HRNC reaction-type prompt, runs a local Hugging Face VLM backend,
and writes one JSON result per sequence for later human review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, LOOKINGFACE_DIR, REACTION_ANNOTATIONS_DIR
from manifest import load_manifest

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "HRNC_reaction_type_vlm_prompt_template.md"
PROMPT_BLOCK_RE = re.compile(r"## Copy-Ready VLM Prompt\s+```text\n(.*?)\n```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_REACTION_CODES = tuple("ABCDEFGHI")
VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class ClipSample:
    seq_id: str
    left_mp4: str
    right_mp4: str
    content_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate LookingFace reaction clips with a local VLM")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local model path")
    parser.add_argument("--backend", choices=["qwen2_5_vl"], default="qwen2_5_vl",
                        help="Inference backend. Only qwen2_5_vl is implemented in this pilot")
    parser.add_argument("--device", default=DEVICE, help="Torch device or device_map setting")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                        help="Model load dtype")
    parser.add_argument("--output_dir", default=REACTION_ANNOTATIONS_DIR,
                        help="Directory for per-clip annotation JSON files")
    parser.add_argument("--fps", type=float, default=2.0, help="Video sampling FPS passed to the VLM")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens per clip")
    parser.add_argument("--pilot", action="store_true",
                        help="Use pilot sampling mode. If neither --pilot nor --all is set, pilot mode is used")
    parser.add_argument("--all", action="store_true", help="Annotate all eligible clips instead of pilot sampling")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of clips to annotate. Defaults to 24 in pilot mode and unlimited with --all")
    parser.add_argument("--categories", nargs="*",
                        choices=["documentary", "game", "movie", "music", "sports", "talk_show"],
                        help="Restrict annotation to specific LookingFace categories")
    parser.add_argument("--seq_ids", nargs="*", help="Restrict annotation to explicit seq_id values")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation JSON files")
    parser.add_argument("--manifest_rebuild", action="store_true", help="Force manifest rebuild before scanning")
    parser.add_argument("--dry_run", action="store_true", help="Print selected clips and exit without inference")
    return parser.parse_args()


def load_prompt_template(prompt_path: Path) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    match = PROMPT_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"Could not find copy-ready VLM prompt in {prompt_path}")
    return match.group(1).strip()


def derive_content_type(video_path: str) -> str:
    relative = Path(video_path).resolve().relative_to(Path(LOOKINGFACE_DIR).resolve())
    if not relative.parts:
        raise ValueError(f"Unable to derive content type from {video_path}")
    category = relative.parts[0]
    if category == "talk_show":
        return "talk show"
    return category


def discover_samples(rebuild_manifest: bool = False) -> list[ClipSample]:
    manifest = load_manifest(rebuild=rebuild_manifest)
    samples: list[ClipSample] = []
    for seq_id, entry in sorted(manifest.items()):
        left_mp4 = entry.get("left_mp4")
        right_mp4 = entry.get("right_mp4")
        if not left_mp4 or not right_mp4:
            continue
        if not os.path.exists(left_mp4) or not os.path.exists(right_mp4):
            continue
        samples.append(
            ClipSample(
                seq_id=seq_id,
                left_mp4=left_mp4,
                right_mp4=right_mp4,
                content_type=derive_content_type(left_mp4),
            )
        )
    return samples


def filter_samples(samples: list[ClipSample], categories: list[str] | None, seq_ids: list[str] | None) -> list[ClipSample]:
    category_filter = None
    if categories:
        category_filter = {category.replace("_", " ") for category in categories}
    seq_id_filter = set(seq_ids) if seq_ids else None

    filtered = []
    for sample in samples:
        if category_filter and sample.content_type not in category_filter:
            continue
        if seq_id_filter and sample.seq_id not in seq_id_filter:
            continue
        filtered.append(sample)
    return filtered


def stratified_pilot_samples(samples: list[ClipSample], limit: int) -> list[ClipSample]:
    if limit <= 0 or len(samples) <= limit:
        return samples[:max(limit, 0)] if limit > 0 else []

    by_category: dict[str, list[ClipSample]] = defaultdict(list)
    for sample in samples:
        by_category[sample.content_type].append(sample)

    ordered_categories = sorted(by_category)
    base_quota = max(1, math.floor(limit / max(len(ordered_categories), 1)))
    selected: list[ClipSample] = []

    for category in ordered_categories:
        bucket = by_category[category]
        selected.extend(bucket[:base_quota])

    if len(selected) >= limit:
        return selected[:limit]

    used_ids = {sample.seq_id for sample in selected}
    remaining: list[ClipSample] = []
    for category in ordered_categories:
        remaining.extend([sample for sample in by_category[category] if sample.seq_id not in used_ids])

    selected.extend(remaining[:limit - len(selected)])
    return selected[:limit]


def render_prompt(prompt_template: str, sample: ClipSample) -> str:
    metadata = (
        f"\n\nClip metadata:\n"
        f"- clip_id: {sample.seq_id}\n"
        f"- content_type: {sample.content_type}\n"
        f"- optional_context: \n"
        f"Return JSON for clip_id {sample.seq_id}."
    )
    return f"{prompt_template}{metadata}"


def dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


class QwenReactionAnnotator:
    def __init__(self, model_id: str, device: str, dtype_name: str, fps: float, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype_name
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_from_name(dtype_name),
            "trust_remote_code": True,
        }
        if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model.to(torch.device(device))
        self.model.eval()

    def annotate(self, sample: ClipSample, prompt_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{sample.left_mp4}", "fps": self.fps},
                    {"type": "video", "video": f"file://{sample.right_mp4}", "fps": self.fps},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def normalize_code_list(values: Any, field_name: str) -> list[str]:
    if values in (None, ""):
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a JSON list")
    normalized = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} entries must be strings")
        code = value.strip().upper()
        if code not in VALID_REACTION_CODES:
            raise ValueError(f"Invalid reaction type code in {field_name}: {value}")
        normalized.append(code)
    return normalized


def parse_json_response(raw_text: str) -> dict[str, Any]:
    candidate = raw_text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(candidate)
        if not match:
            raise ValueError("Model output did not contain a JSON object")
        return json.loads(match.group(0))


def validate_annotation(parsed: dict[str, Any], sample: ClipSample) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Parsed annotation must be a JSON object")

    clip_id = parsed.get("clip_id")
    if clip_id is None:
        parsed["clip_id"] = sample.seq_id
    elif str(clip_id) != sample.seq_id:
        raise ValueError(f"clip_id mismatch: expected {sample.seq_id}, got {clip_id}")

    dominant = str(parsed.get("dominant_reaction_type", "")).strip().upper()
    if dominant not in VALID_REACTION_CODES:
        raise ValueError("dominant_reaction_type must be one of A-I")

    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in VALID_CONFIDENCE:
        raise ValueError("confidence must be one of high, medium, low")

    visual_evidence = str(parsed.get("visual_evidence", "")).strip()
    stimulus_relation = str(parsed.get("stimulus_relation", "")).strip()
    uncertainty_reason = str(parsed.get("uncertainty_reason", "")).strip()
    if not visual_evidence:
        raise ValueError("visual_evidence is required")
    if not stimulus_relation:
        raise ValueError("stimulus_relation is required")
    if (dominant == "I" or confidence == "low") and not uncertainty_reason:
        raise ValueError("uncertainty_reason is required when confidence is low or dominant_reaction_type is I")

    return {
        "clip_id": sample.seq_id,
        "dominant_reaction_type": dominant,
        "co_occurring_reaction_types": normalize_code_list(parsed.get("co_occurring_reaction_types", []), "co_occurring_reaction_types"),
        "transition_order": normalize_code_list(parsed.get("transition_order", []), "transition_order"),
        "visual_evidence": visual_evidence,
        "stimulus_relation": stimulus_relation,
        "confidence": confidence,
        "uncertainty_reason": uncertainty_reason,
    }


def output_path_for(output_dir: str, sample: ClipSample) -> str:
    return os.path.join(output_dir, f"{sample.seq_id}.json")


def build_result_record(
    sample: ClipSample,
    annotation: dict[str, Any] | None,
    raw_response: str,
    prompt_text: str,
    model_id: str,
    backend: str,
    error: str | None = None,
) -> dict[str, Any]:
    record = {
        "status": "ok" if error is None else "error",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "backend": backend,
        "prompt_template": str(PROMPT_TEMPLATE_PATH),
        "prompt_text": prompt_text,
        "sample": asdict(sample),
        "raw_response": raw_response,
    }
    if annotation is not None:
        record["annotation"] = annotation
    if error is not None:
        record["error"] = error
    return record


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    use_pilot = args.pilot or not args.all
    limit = 24 if use_pilot and args.limit is None else args.limit
    samples = discover_samples(rebuild_manifest=args.manifest_rebuild)
    samples = filter_samples(samples, args.categories, args.seq_ids)
    if use_pilot:
        samples = stratified_pilot_samples(samples, limit if limit is not None else 24)
    elif limit is not None and limit > 0:
        samples = samples[:limit]

    if not samples:
        print("No eligible clips found for annotation.")
        return

    print(f"Selected {len(samples)} clips for annotation.")
    for sample in samples:
        print(f"  {sample.seq_id}  [{sample.content_type}]")
    if args.dry_run:
        return

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    annotator = QwenReactionAnnotator(
        model_id=args.model_id,
        device=args.device,
        dtype_name=args.dtype,
        fps=args.fps,
        max_new_tokens=args.max_new_tokens,
    )

    completed = 0
    skipped = 0
    failed = 0

    for index, sample in enumerate(samples, start=1):
        destination = output_path_for(args.output_dir, sample)
        if os.path.exists(destination) and not args.overwrite:
            print(f"[{index}/{len(samples)}] Skipping {sample.seq_id}: cached")
            skipped += 1
            continue

        prompt_text = render_prompt(prompt_template, sample)
        print(f"[{index}/{len(samples)}] Annotating {sample.seq_id} [{sample.content_type}]")
        try:
            raw_response = annotator.annotate(sample, prompt_text)
            parsed = parse_json_response(raw_response)
            annotation = validate_annotation(parsed, sample)
            record = build_result_record(
                sample=sample,
                annotation=annotation,
                raw_response=raw_response,
                prompt_text=prompt_text,
                model_id=args.model_id,
                backend=args.backend,
            )
            save_json(destination, record)
            completed += 1
        except Exception as exc:
            raw_response = locals().get("raw_response", "")
            record = build_result_record(
                sample=sample,
                annotation=None,
                raw_response=raw_response,
                prompt_text=prompt_text,
                model_id=args.model_id,
                backend=args.backend,
                error=str(exc),
            )
            save_json(destination, record)
            failed += 1
            print(f"  ERROR: {exc}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done: {completed} completed, {skipped} skipped, {failed} failed")
    print(f"Annotations written to: {args.output_dir}")


if __name__ == "__main__":
    main()