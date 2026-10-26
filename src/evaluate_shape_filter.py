#!/usr/bin/env python3
"""Evaluate class-specific score and box-shape filters on a cached public set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"))

from detection_eval import evaluate_3d_detection  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--class-name", choices=("vehicle", "pedestrian", "truck"), required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.03, 0.05, 0.1, 0.2])
    parser.add_argument("--max-lengths", type=float, nargs="+", default=[100.0])
    parser.add_argument("--min-length", type=float, default=0.0)
    parser.add_argument("--max-width", type=float, default=100.0)
    parser.add_argument("--min-width", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    source = cache["predictions"]
    ground_truth = cache["ground_truth"]
    results = []
    for threshold in args.thresholds:
        for max_length in args.max_lengths:
            predictions = {}
            for frame_id, rows in source.items():
                predictions[frame_id] = [
                    row
                    for row in rows
                    if row.get("class", "vehicle").lower() == args.class_name
                    and float(row["score"]) >= threshold
                    and args.min_length <= float(row["box"][3]) <= max_length
                    and args.min_width <= float(row["box"][4]) <= args.max_width
                ]
            result = evaluate_3d_detection(
                predictions, ground_truth, [args.class_name], (0.5,), "3d"
            )
            results.append({
                "threshold": threshold,
                "min_length": args.min_length,
                "max_length": max_length,
                "min_width": args.min_width,
                "max_width": args.max_width,
                "AP_0.5_R40": result["mAP_0.5_R40"],
                "AP_0.5_R11": result["mAP_0.5_R11"],
                "n_gt": result["n_gt"],
                "n_pred": result["n_pred"],
            })
    results.sort(key=lambda row: row["AP_0.5_R40"], reverse=True)
    payload = {"class": args.class_name, "cache": str(args.cache), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
