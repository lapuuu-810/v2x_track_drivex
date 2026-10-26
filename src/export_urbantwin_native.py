#!/usr/bin/env python3
"""Export official UrbanTwin simulated scans in the challenge XYZ contract.

The source is the released UT-V2X-Real-IC digital-twin dataset, not a real
V2X-Real frame.  Files are range-masked and deterministically subsampled only
to meet the competition's per-file size cap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="UrbanTwin pcd directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.frames != 50:
        raise ValueError("the competition package requires exactly 50 scans")
    if args.points_per_frame * 12 > 1_500_000:
        raise ValueError("points-per-frame exceeds the 1.5 MB file-size cap")
    files = sorted(args.source.glob("*.npy"))
    if len(files) < args.frames:
        raise FileNotFoundError(f"expected at least {args.frames} .npy scans in {args.source}")
    selected = [files[index] for index in np.linspace(0, len(files) - 1, args.frames, dtype=int)]
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(selected, 1):
        points = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
        points = points[valid]
        if len(points) < args.points_per_frame:
            raise ValueError(f"{path} has only {len(points)} in-range points")
        output = points[rng.choice(len(points), args.points_per_frame, replace=False)]
        output.astype(np.float32).tofile(args.output / f"{index:04d}.bin")
        rows.append({"source": path.name, "in_range_points": int(len(points)), "output_points": args.points_per_frame})
    manifest = {
        "method": "released UT-V2X-Real-IC digital-twin native LiDAR export",
        "source": str(args.source),
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "range": [*LOW.tolist(), *HIGH.tolist()],
        "seed": args.seed,
        "scans": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "scans"}, indent=2))


if __name__ == "__main__":
    main()
