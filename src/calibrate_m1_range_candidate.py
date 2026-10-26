#!/usr/bin/env python3
"""Constrain generated M1 ranges using an independently fitted sensor envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bounds(corpus: Path, low: float, high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((corpus / "train").glob("*.npz"))
    rows = [[] for _ in range(64)]
    for path in files:
        with np.load(path) as item:
            log_range, mask = item["log_range"].astype(np.float32), item["mask"].astype(bool)
        value = np.exp2(log_range * 8.0) - 1.0
        for row in range(64):
            if mask[row].any():
                rows[row].append(value[row, mask[row]])
    lo = np.asarray([np.quantile(np.concatenate(row), low) for row in rows], np.float32)
    hi = np.asarray([np.quantile(np.concatenate(row), high) for row in rows], np.float32)
    return np.load(corpus / "ring_angles.npy").astype(np.float32), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.002)
    parser.add_argument("--high", type=float, default=0.995)
    args = parser.parse_args()
    angles, lo, hi = bounds(args.corpus, args.low, args.high)
    source, target = args.input / "synthetic", args.output / "synthetic"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(source.glob("*.bin")):
        xyz = np.fromfile(path, np.float32).reshape(-1, 3)
        radius = np.linalg.norm(xyz, axis=1)
        elevation = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
        row = np.abs(elevation[:, None] - angles[None]).argmin(1)
        radius = np.clip(radius, lo[row], hi[row])
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        out = np.column_stack((radius * np.cos(elevation) * np.cos(azimuth), radius * np.cos(elevation) * np.sin(azimuth), radius * np.sin(elevation))).astype(np.float32)
        out.tofile(target / path.name)
        records.append({"frame": path.name, "points": int(len(out)), "range_min": float(radius.min()), "range_max": float(radius.max())})
    report = {"input": str(args.input), "low": args.low, "high": args.high, "frames": len(records), "row_range_low": lo.tolist(), "row_range_high": hi.tolist(), "records": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"records", "row_range_low", "row_range_high"}}, indent=2))


if __name__ == "__main__":
    main()
