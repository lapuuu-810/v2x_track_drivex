#!/usr/bin/env python3
"""Rank explicit per-class box-calibration candidates on a cached public set."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORING_ROOT = PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"
sys.path.insert(0, str(SCORING_ROOT))

from detection_eval import evaluate_3d_detection  # noqa: E402


def parse_profile(value):
    name, separator, raw = value.partition("=")
    numbers = tuple(float(part) for part in raw.split(",")) if separator else ()
    if not name or len(numbers) != 4 or any(scale <= 0 for scale in numbers[:3]):
        raise ValueError(f"Profile must be NAME=DX,DY,DZ,Z_OFFSET: {value}")
    return name, numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--class-name", choices=("vehicle", "pedestrian", "truck"), required=True)
    parser.add_argument("--profile", action="append", default=[], metavar="NAME=DX,DY,DZ,Z")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1, 0.2])
    parser.add_argument("--fast", action="store_true", help="Evaluate only 3D AP at IoU 0.5.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    source_predictions = cache["predictions"]
    ground_truth = cache["ground_truth"]
    profiles = dict(parse_profile(value) for value in args.profile)
    if not profiles:
        profiles = {"identity": (1.0, 1.0, 1.0, 0.0)}

    results = []
    for (profile_name, (sx, sy, sz, z_offset)), threshold in itertools.product(
        profiles.items(), args.thresholds
    ):
        predictions = {}
        for frame_id, rows in source_predictions.items():
            selected = []
            for row in rows:
                if row.get("class", "vehicle").lower() != args.class_name:
                    continue
                if float(row["score"]) < threshold:
                    continue
                calibrated = copy.deepcopy(row)
                calibrated["class"] = args.class_name
                calibrated["box"][2] += z_offset
                calibrated["box"][3] *= sx
                calibrated["box"][4] *= sy
                calibrated["box"][5] *= sz
                selected.append(calibrated)
            predictions[frame_id] = selected

        thresholds = (0.5,) if args.fast else (0.5, 0.7)
        result_3d = evaluate_3d_detection(
            predictions, ground_truth, [args.class_name], thresholds, "3d"
        )
        result_bev = None if args.fast else evaluate_3d_detection(
            predictions, ground_truth, [args.class_name], thresholds, "bev"
        )
        results.append({
            "profile": profile_name,
            "scale": [sx, sy, sz],
            "z_offset": z_offset,
            "threshold": threshold,
            "3d_05_R40": result_3d["mAP_0.5_R40"],
            "3d_07_R40": result_3d.get("mAP_0.7_R40"),
            "bev_05_R40": result_bev["mAP_0.5_R40"] if result_bev is not None else None,
            "n_gt": result_3d["n_gt"],
            "n_pred": result_3d["n_pred"],
        })

    results.sort(
        key=lambda row: (row["3d_05_R40"], row["3d_07_R40"] or -1.0, row["bev_05_R40"] or -1.0),
        reverse=True,
    )
    payload = {"class": args.class_name, "cache": str(args.cache), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
