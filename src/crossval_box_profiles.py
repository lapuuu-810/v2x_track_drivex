#!/usr/bin/env python3
"""Report per-scenario AP for a small set of box calibration profiles."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition/starting_kit/scoring_program"))
from detection_eval import evaluate_3d_detection  # noqa: E402
from evaluate_box_calibration import parse_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text())
    profiles = dict(parse_profile(value) for value in args.profile)
    scenarios = sorted({frame_id.split("_", 1)[0] for frame_id in cache["ground_truth"]})
    rows = []
    for scenario in scenarios:
        ids = {
            frame_id for frame_id in cache["ground_truth"]
            if frame_id.split("_", 1)[0] == scenario
        }
        ground_truth = {key: cache["ground_truth"][key] for key in ids}
        for name, (sx, sy, sz, z_offset) in profiles.items():
            predictions = {}
            for frame_id in ids:
                selected = []
                for source in cache["predictions"][frame_id]:
                    if source["class"].lower() != args.class_name:
                        continue
                    if float(source["score"]) < args.threshold:
                        continue
                    row = copy.deepcopy(source)
                    row["box"][2] += z_offset
                    row["box"][3] *= sx
                    row["box"][4] *= sy
                    row["box"][5] *= sz
                    selected.append(row)
                predictions[frame_id] = selected
            result = evaluate_3d_detection(
                predictions, ground_truth, [args.class_name], (0.5,), "3d"
            )
            rows.append({
                "scenario": scenario,
                "profile": name,
                "frames": len(ids),
                "n_gt": result["n_gt"],
                "n_pred": result["n_pred"],
                "AP_0.5_R40": result["mAP_0.5_R40"],
            })
    payload = {"class": args.class_name, "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
