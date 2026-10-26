#!/usr/bin/env python3
"""Quantize prediction values to the compact format used by submission v17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-decimals", type=int, default=3)
    parser.add_argument("--score-decimals", type=int, default=5)
    args = parser.parse_args()

    predictions = json.loads(args.input.read_text())
    for rows in predictions.values():
        for row in rows:
            row["box"] = [round(float(value), args.box_decimals) for value in row["box"]]
            row["score"] = round(float(row["score"]), args.score_decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions))
    print(json.dumps({
        "frames": len(predictions),
        "detections": sum(map(len, predictions.values())),
        "box_decimals": args.box_decimals,
        "score_decimals": args.score_decimals,
    }))


if __name__ == "__main__":
    main()
