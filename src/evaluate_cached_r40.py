#!/usr/bin/env python3
"""Evaluate cached V2X-Real predictions with the challenge R40 metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "starting_kit" / "scoring_program"))
from detection_eval import evaluate_3d_detection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classes = ["vehicle", "pedestrian", "truck"]
    rows = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        predictions = {
            frame_id: [item for item in items if item["score"] >= args.threshold]
            for frame_id, items in data["predictions"].items()
        }
        d3 = evaluate_3d_detection(predictions, data["ground_truth"], classes, (0.5, 0.7), "3d")
        bev = evaluate_3d_detection(predictions, data["ground_truth"], classes, (0.5, 0.7), "bev")
        rows.append({
            "input": str(path),
            "checkpoint": data.get("checkpoint"),
            "threshold": args.threshold,
            "n_gt": d3["n_gt"],
            "n_pred": d3["n_pred"],
            "3d_05_R40": d3["mAP_0.5_R40"],
            "3d_07_R40": d3["mAP_0.7_R40"],
            "bev_05_R40": bev["mAP_0.5_R40"],
            "per_class": d3["per_class"],
        })
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
