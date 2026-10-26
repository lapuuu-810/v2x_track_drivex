#!/usr/bin/env python3
"""Deterministically turn generated XYZ frames into submission-style sparse bins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LIMITS = np.asarray([[-40., 40.], [-40., 40.], [-8., 2.]], np.float32)


def read(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        x = np.asarray(np.load(path), np.float32).reshape(-1, 3)
    else:
        x = np.fromfile(path, np.float32).reshape(-1, 3)
    keep = np.isfinite(x).all(1) & ((x >= LIMITS[:, 0]) & (x <= LIMITS[:, 1])).all(1)
    return x[keep]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--points", type=int, default=9000)
    p.add_argument("--seed", type=int, default=20260728)
    a = p.parse_args()
    files = sorted(a.input.glob("*.npy")) or sorted(a.input.glob("*.bin"))
    if len(files) != 50:
        raise SystemExit(f"expected exactly 50 frames, got {len(files)}")
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    for i, file in enumerate(files, 1):
        x = read(file)
        if len(x) < a.points:
            raise ValueError(f"{file} has {len(x)} valid points")
        rng = np.random.default_rng(a.seed + i)
        y = x[rng.choice(len(x), a.points, replace=False)]
        y.astype(np.float32).tofile(a.output / f"{i:04d}.bin")
        report.append({"source": file.name, "valid": len(x), "output": len(y)})
    (a.output / "manifest.json").write_text(json.dumps({"points": a.points, "seed": a.seed, "frames": report}, indent=2) + "\n")
    print(json.dumps({"frames": len(report), "points_per_frame": a.points, "valid_median": int(np.median([x['valid'] for x in report]))}))


if __name__ == "__main__":
    main()
