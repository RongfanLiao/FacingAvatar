#!/usr/bin/env python3
"""Compare benchmark metrics across multiple ported-model runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


DEFAULT_METRICS = [
    "mae",
    "rmse",
    "frcorr",
    "frdist",
    "delta_mae",
    "delta_rmse",
    "fid_delta_fm",
    "snd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark metrics across multiple checkpoint runs")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Checkpoint directories or metrics.json files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching the provided runs",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric keys to display; defaults to the shared core benchmark metrics",
    )
    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show all keys shared by every run instead of the default core metric set",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "markdown", "csv"],
        default="plain",
        help="Output format for the comparison table",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path for the rendered table",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Floating-point precision for rendered metric values",
    )
    return parser.parse_args()


def resolve_metrics_path(run_path: str) -> Path:
    path = Path(run_path)
    if path.is_dir():
        metrics_path = path / "metrics.json"
    else:
        metrics_path = path

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found for run: {run_path}")
    return metrics_path


def load_run(run_path: str, label: str | None) -> dict[str, object]:
    metrics_path = resolve_metrics_path(run_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    inferred_label = label or metrics_path.parent.name or metrics_path.stem
    return {
        "label": inferred_label,
        "path": str(metrics_path),
        "metrics": metrics,
    }


def infer_metric_keys(runs: list[dict[str, object]], args: argparse.Namespace) -> list[str]:
    if args.metrics:
        return list(args.metrics)

    if args.show_all:
        shared = None
        for run in runs:
            keys = set(run["metrics"].keys())
            shared = keys if shared is None else shared & keys
        assert shared is not None
        return sorted(key for key in shared if key not in {"target_variant"})

    return [key for key in DEFAULT_METRICS if all(key in run["metrics"] for run in runs)]


def format_value(value: object, precision: int) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "-"
    return str(value)


def build_rows(runs: list[dict[str, object]], metric_keys: list[str], precision: int) -> tuple[list[str], list[list[str]]]:
    headers = ["run", "target_variant", *metric_keys]
    rows = []
    for run in runs:
        metrics = run["metrics"]
        row = [
            str(run["label"]),
            format_value(metrics.get("target_variant", "-"), precision),
        ]
        row.extend(format_value(metrics.get(key, "-"), precision) for key in metric_keys)
        rows.append(row)
    return headers, rows


def render_plain(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    sep = "  ".join("-" * width for width in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def render_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def render_csv(headers: list[str], rows: list[list[str]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\n")


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) not in {0, len(args.runs)}:
        raise ValueError("--labels must be omitted or provide exactly one label per run")

    labels = args.labels or [None] * len(args.runs)
    runs = [load_run(run_path, label) for run_path, label in zip(args.runs, labels)]
    metric_keys = infer_metric_keys(runs, args)
    headers, rows = build_rows(runs, metric_keys, precision=args.precision)

    if args.format == "plain":
        rendered = render_plain(headers, rows)
    elif args.format == "markdown":
        rendered = render_markdown(headers, rows)
    else:
        rendered = render_csv(headers, rows)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved comparison table to: {output_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()