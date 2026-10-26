#!/usr/bin/env python3
"""Build a human-sized pedestrian GT database with all required class keys."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("rb") as handle:
        source = pickle.load(handle)
    selected = [
        entry for entry in source.get("pedestrian", [])
        if entry["box3d_lidar"][3] <= 1.2
        and entry["box3d_lidar"][4] <= 1.0
        and 1.35 <= entry["box3d_lidar"][5] <= 2.1
    ]
    with args.output.open("wb") as handle:
        pickle.dump({"vehicle": [], "pedestrian": selected, "truck": []}, handle)
    print(f"Selected {len(selected)} human-sized pedestrians; wrote all three class keys")


if __name__ == "__main__":
    main()
