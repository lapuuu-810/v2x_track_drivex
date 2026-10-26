#!/usr/bin/env python3
"""Evaluate prediction JSON files against labels paired with a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "competition/starting_kit/scoring_program"),
]

from detection_eval import evaluate_3d_detection  # noqa: E402
from eval_v2xreal_public_gblobs import ground_truth_for_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    ground_truth = {}
    for item in manifest:
        point_path = ROOT / item["path"]
        ground_truth[item["id"]] = ground_truth_for_sample(
            {"yaml": point_path.with_suffix(".yaml")}
        )

    classes = ["vehicle", "pedestrian", "truck"]
    expected_ids = set(ground_truth)
    results = []
    for path in sorted(args.prediction_dir.glob(args.pattern)):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != expected_ids:
            continue
        if data and isinstance(next(iter(data.values())), dict):
            continue
        detection_3d = evaluate_3d_detection(
            data, ground_truth, classes, (0.5, 0.7), "3d"
        )
        detection_bev = evaluate_3d_detection(
            data, ground_truth, classes, (0.5, 0.7), "bev"
        )
        row = {
            "file": path.name,
            "n_pred": detection_3d["n_pred"],
            "3d_05": detection_3d["mAP_0.5_R40"],
            "3d_07": detection_3d["mAP_0.7_R40"],
            "bev_05": detection_bev["mAP_0.5_R40"],
            "detection_proxy": (
                detection_3d["mAP_0.5_R40"]
                + detection_3d["mAP_0.7_R40"]
                + detection_bev["mAP_0.5_R40"]
            )
            / 3.0,
            "details_3d": detection_3d,
        }
        results.append(row)
        print(
            f"{path.name:55s} n={row['n_pred']:5d} "
            f"3d5={row['3d_05']:.6f} 3d7={row['3d_07']:.6f} "
            f"bev5={row['bev_05']:.6f} proxy={row['detection_proxy']:.6f}"
        )
    results.sort(key=lambda row: row["detection_proxy"], reverse=True)
    report = {
        "frames": len(ground_truth),
        "n_gt": sum(map(len, ground_truth.values())),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if results:
        print("BEST", results[0]["file"], results[0]["detection_proxy"])


if __name__ == "__main__":
    main()
