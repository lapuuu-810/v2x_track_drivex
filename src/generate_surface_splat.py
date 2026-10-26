#!/usr/bin/env python3
"""Surface-aware LiDAR re-sampling from generated UrbanTwin frames.

Each output return is drawn from a local tangent-plane kernel estimated only
from the input synthetic cloud.  This is a geometry-preserving sensor model:
it neither reads nor replays public/hidden real scans.  The per-frame point
count and file interface remain identical to the starting-kit contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RANGE_LO = np.asarray([-40.0, -40.0, -8.0], np.float32)
RANGE_HI = np.asarray([40.0, 40.0, 2.0], np.float32)


def read_cloud(path: Path) -> np.ndarray:
    x = np.fromfile(path, np.float32)
    if x.size % 3:
        raise ValueError(f"{path}: expected float32 XYZ")
    x = x.reshape(-1, 3)
    keep = np.isfinite(x).all(1) & np.any(x != 0, 1)
    keep &= (x >= RANGE_LO).all(1) & (x <= RANGE_HI).all(1)
    return x[keep]


def splat(points: np.ndarray, sigma: float, neighbours: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one tangent-plane perturbation per input return."""
    k = min(neighbours, len(points))
    _, indices = cKDTree(points).query(points, k=k, workers=-1)
    if indices.ndim == 1:
        indices = indices[:, None]
    patches = points[indices]
    centered = patches - patches.mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    # Tangential spread is proportional to local point spacing but bounded;
    # the smallest eigenvector is the estimated surface normal and is unused.
    tangent_scale = np.sqrt(np.maximum(values[:, 1:], 1e-8))
    tangent_scale = np.clip(tangent_scale, 0.02, 0.35)
    noise = rng.normal(size=(len(points), 2)).astype(np.float32)
    delta = np.einsum("nji,ni->nj", vectors[:, :, 1:], noise * tangent_scale * sigma)
    return np.clip(points + delta, RANGE_LO, RANGE_HI).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sigma", type=float, required=True, help="dimensionless local tangent-kernel scale")
    p.add_argument("--neighbours", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260729)
    a = p.parse_args()
    files = sorted(a.input.glob("*.bin"))
    if len(files) != 50:
        raise SystemExit(f"expected exactly 50 input frames, got {len(files)}")
    if not 0.0 < a.sigma <= 1.0:
        raise SystemExit("--sigma must be in (0, 1]")
    a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    counts = []
    for i, source in enumerate(files, 1):
        out = splat(read_cloud(source), a.sigma, a.neighbours, rng)
        out.tofile(a.output / f"{i:04d}.bin")
        counts.append(int(len(out)))
    manifest = {
        "method": "local tangent-plane surface splatting from synthetic UrbanTwin frames",
        "real_data_used": False,
        "input": str(a.input),
        "sigma": a.sigma,
        "neighbours": a.neighbours,
        "seed": a.seed,
        "frames": len(files),
        "point_counts": counts,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "point_counts"}, indent=2))


if __name__ == "__main__":
    main()
