#!/usr/bin/env python3
"""Deterministically compare local synthetic LiDAR point-cloud populations.

This utility deliberately accepts only locally generated candidate scans.  It
never reads competition test/reference data, so it can be used to diagnose why
two legal submissions have different global point statistics before spending a
submission attempt.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sources must be LABEL=PATH")
    label, raw = value.split("=", 1)
    if not label or not raw:
        raise argparse.ArgumentTypeError("sources must be LABEL=PATH")
    path = Path(raw)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"missing source: {path}")
    return label, path


def load_folder(root: Path) -> list[np.ndarray]:
    files = sorted(root.glob("*.bin"))
    if not files and (root / "synthetic").is_dir():
        files = sorted((root / "synthetic").glob("*.bin"))
    if not files:
        raise ValueError(f"no .bin scans in {root}")
    return [np.fromfile(path, dtype=np.float32).reshape(-1, 3) for path in files]


def load_zip(path: Path) -> list[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("synthetic/") and name.endswith(".bin"))
        if not names:
            raise ValueError(f"no synthetic/*.bin scans in {path}")
        return [np.frombuffer(archive.read(name), dtype=np.float32).reshape(-1, 3).copy() for name in names]


def reservoir_sample(scans: list[np.ndarray], limit: int, seed: int) -> np.ndarray:
    """Uniform point sample independent of scan ordering, with bounded RAM."""
    rng = np.random.default_rng(seed)
    sample = np.empty((limit, 3), dtype=np.float32)
    seen = 0
    kept = 0
    for scan in scans:
        for start in range(0, len(scan), 65_536):
            block = scan[start:start + 65_536]
            if kept < limit:
                take = min(limit - kept, len(block))
                sample[kept:kept + take] = block[:take]
                kept += take
                block = block[take:]
                seen += take
            if len(block):
                positions = rng.integers(0, seen + len(block), size=len(block))
                replace = positions < limit
                sample[positions[replace]] = block[replace]
                seen += len(block)
    return sample[:kept]


def summary(scans: list[np.ndarray], limit: int, seed: int) -> dict:
    counts = np.asarray([len(scan) for scan in scans], dtype=np.int64)
    points = reservoir_sample(scans, limit, seed)
    radial = np.linalg.norm(points[:, :2], axis=1)
    xyz_q = np.quantile(points, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    r_q = np.quantile(radial, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    bins = np.linspace(0, 80, 17)
    radial_hist = np.histogram(radial, bins=bins)[0].astype(float)
    radial_hist = (radial_hist / radial_hist.sum()).tolist()
    return {
        "frames": int(len(scans)),
        "total_points": int(counts.sum()),
        "points_per_frame": {"mean": float(counts.mean()), "std": float(counts.std()), "min": int(counts.min()), "max": int(counts.max())},
        "sampled_points": int(len(points)),
        "xyz_mean": points.mean(axis=0).astype(float).tolist(),
        "xyz_std": points.std(axis=0).astype(float).tolist(),
        "xyz_quantiles_1_5_50_95_99": xyz_q.astype(float).tolist(),
        "xyz_covariance": np.cov(points.T).astype(float).tolist(),
        "xy_radius_quantiles_1_5_25_50_75_95_99": r_q.astype(float).tolist(),
        "xy_radius_histogram_0_80m_5m_bins": radial_hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=parse_source, required=True, help="LABEL=DIR_OR_ZIP; repeatable")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_points < 1:
        raise ValueError("--max-points must be positive")
    report: dict[str, object] = {"protocol": "local candidate-only distribution audit", "sources": {}}
    for index, (label, path) in enumerate(args.source):
        scans = load_zip(path) if path.suffix == ".zip" else load_folder(path)
        report["sources"][label] = summary(scans, args.max_points, args.seed + index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
