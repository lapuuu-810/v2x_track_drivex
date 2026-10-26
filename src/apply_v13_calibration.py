#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


PROFILES = {
    "vehicle": ((1.0, 1.10, 1.15), 0.10),
    "pedestrian": ((1.0, 1.0, 1.0), 0.0),
    "truck": ((1.20, 1.10, 1.0), 0.15),
}


def calibrate(predictions: dict) -> dict:
    output = {}
    for frame_id, rows in predictions.items():
        selected = []
        for source in rows:
            row = copy.deepcopy(source)
            name = row["class"].lower()
            # The synthetic model produces many car-sized false pedestrian boxes.
            if name == "pedestrian" and (row["box"][3] > 0.7 or row["box"][4] > 1.2):
                continue
            scale, z_offset = PROFILES[name]
            row["box"][2] += z_offset
            for index, value in enumerate(scale, start=3):
                row["box"][index] *= value
            selected.append(row)
        output[frame_id] = selected
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source = payload.get("predictions", payload)
    predictions = calibrate(source)
    output = copy.deepcopy(payload) if "predictions" in payload else predictions
    if isinstance(output, dict) and "predictions" in output:
        output["predictions"] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    report = {
        "frames": len(predictions),
        "detections": sum(len(rows) for rows in predictions.values()),
        "profiles": PROFILES,
    }
    if args.score:
        if "ground_truth" not in payload:
            raise ValueError("--score requires a cache containing ground_truth")
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "competition/starting_kit/scoring_program"))
        from detection_eval import evaluate_3d_detection

        classes = ["vehicle", "pedestrian", "truck"]
        for metric in ("3d", "bev"):
            result = evaluate_3d_detection(
                predictions, payload["ground_truth"], classes, (0.5, 0.7), metric
            )
            report[metric] = {
                "mAP_0.5_R40": result["mAP_0.5_R40"],
                "mAP_0.7_R40": result["mAP_0.7_R40"],
                "per_class_0.5_R40": {
                    name: result["per_class"][name]["AP_0.5_R40"] for name in classes
                },
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
