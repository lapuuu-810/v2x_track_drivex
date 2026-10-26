#!/usr/bin/env python3
"""Deterministically reduce a valid synthetic submission scan set.

This is a storage-only transform: every output point is selected from its
corresponding generated synthetic scan; no real data is read and no geometry is
altered.  The result is useful where a platform account has a tight upload
quota, while preserving the aggregate distribution as closely as possible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-frame", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.bin"))
    if len(paths) != 50:
        raise ValueError(f"expected exactly 50 input scans, got {len(paths)}")
    if args.points_per_frame < 1 or args.points_per_frame * 12 > 1_500_000:
        raise ValueError("invalid output point count")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 3:
            raise ValueError(f"invalid XYZ input: {path}")
        cloud = raw.reshape(-1, 3)
        rng = np.random.default_rng(args.seed + index)
        chosen = cloud[rng.choice(len(cloud), args.points_per_frame, replace=len(cloud) < args.points_per_frame)]
        chosen.astype(np.float32).tofile(args.output / f"{index:04d}.bin")
        rows.append({"source": path.name, "input_points": len(cloud), "output_points": len(chosen)})
    (args.output / "manifest.json").write_text(json.dumps({
        "method": "deterministic per-scan subsampling of bounded-SOT96 synthetic scans",
        "input": str(args.input), "frames": 50,
        "points_per_frame": args.points_per_frame, "seed": args.seed,
        "real_data_read": False, "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": 50, "points_per_frame": args.points_per_frame, "output": str(args.output)}))


if __name__ == "__main__":
    main()
