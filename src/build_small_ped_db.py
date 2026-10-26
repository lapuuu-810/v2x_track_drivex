#!/usr/bin/env python3
"""Build a GT-sampling database restricted to human-sized pedestrians."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=float, default=1.2)
    parser.add_argument("--max-width", type=float, default=1.0)
    parser.add_argument("--min-height", type=float, default=1.35)
    parser.add_argument("--max-height", type=float, default=2.1)
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        db = pickle.load(handle)
    selected = []
    for entry in db.get("pedestrian", []):
        box = entry["box3d_lidar"]
        if (
            box[3] <= args.max_length
            and box[4] <= args.max_width
            and args.min_height <= box[5] <= args.max_height
        ):
            selected.append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump({"pedestrian": selected}, handle)
    print(f"Selected {len(selected)} human-sized pedestrians from {len(db.get('pedestrian', []))}")


if __name__ == "__main__":
    main()
