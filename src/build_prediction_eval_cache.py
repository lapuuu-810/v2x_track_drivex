#!/usr/bin/env python3
"""Attach manifest-derived public ground truth to cached predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_v2xreal_public_gblobs import ground_truth_for_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    predictions = json.loads(args.predictions.read_text())
    ground_truth = {}
    for item in manifest:
        point_path = Path(item["path"])
        if not point_path.is_absolute():
            point_path = ROOT / point_path
        ground_truth[item["id"]] = ground_truth_for_sample(
            {"yaml": point_path.with_suffix(".yaml")}
        )
    if set(predictions) != set(ground_truth):
        raise ValueError("Prediction and ground-truth frame IDs differ")
    payload = {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "frames": len(ground_truth),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(json.dumps({
        "frames": len(ground_truth),
        "n_pred": sum(map(len, predictions.values())),
        "n_gt": sum(map(len, ground_truth.values())),
    }))


if __name__ == "__main__":
    main()
