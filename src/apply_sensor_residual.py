#!/usr/bin/env python3
"""Apply small deterministic LiDAR sensor residuals without changing point count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOUNDS = np.asarray([[-40.0, 40.0], [-40.0, 40.0], [-8.0, 2.0]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radial-sigma", type=float, default=0.0,
                        help="base radial range noise in metres")
    parser.add_argument("--range-slope", type=float, default=0.0,
                        help="additional radial sigma per metre of range")
    parser.add_argument("--x-quadratic", type=float, default=0.0,
                        help="x scale residual: 1 - q*(abs(x)/40)^2")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    files = sorted(args.input.glob("*.bin"))
    if len(files) != 50:
        raise SystemExit(f"expected 50 XYZ frames, got {len(files)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        points = np.fromfile(path, np.float32).reshape(-1, 3).copy()
        if args.x_quadratic:
            points[:, 0] *= 1.0 - args.x_quadratic * (np.abs(points[:, 0]) / 40.0) ** 2
        if args.radial_sigma or args.range_slope:
            rng = np.random.default_rng(args.seed + index * 1009)
            radius = np.linalg.norm(points, axis=1)
            sigma = args.radial_sigma + args.range_slope * radius
            points += points / np.maximum(radius[:, None], 1e-6) * rng.normal(0.0, sigma)[:, None]
        points = np.clip(points, BOUNDS[:, 0], BOUNDS[:, 1]).astype(np.float32)
        points.tofile(args.output / f"{index:04d}.bin")
    (args.output / "manifest.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(json.dumps({"frames": len(files), "radial_sigma": args.radial_sigma,
                      "range_slope": args.range_slope, "x_quadratic": args.x_quadratic}))


if __name__ == "__main__":
    main()
