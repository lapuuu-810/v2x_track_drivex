#!/usr/bin/env python3
"""Extract an unlabeled roadside-LiDAR sensor profile from allowed real -1 frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RANGE_EDGES = np.arange(0, 45, 5, dtype=np.float32)
AZIMUTH_EDGES = np.linspace(-np.pi, np.pi, 73, dtype=np.float32)


def load_xyz(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    width = 4 if raw.size % 4 == 0 else 3
    points = raw.reshape(-1, width)[:, :3]
    mask = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    radius = np.linalg.norm(points, axis=1)
    return points[mask & (radius > 0.5) & (radius < 80)]


def fit_rings(elevations: np.ndarray, count: int) -> np.ndarray:
    centers = np.quantile(elevations, (np.arange(count) + 0.5) / count)
    for _ in range(30):
        assignment = np.abs(elevations[:, None] - centers[None]).argmin(axis=1)
        updated = np.asarray([
            np.median(elevations[assignment == i]) if np.any(assignment == i) else centers[i]
            for i in range(count)
        ])
        if np.max(np.abs(updated - centers)) < 1e-7:
            break
        centers = updated
    return np.sort(centers).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rings", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--points-per-frame", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    forbidden = {
        line.split("_")[-1].strip()
        for line in args.forbidden.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    paths = sorted(args.real_root.glob("train*/**/-1/*.bin"))
    paths = [path for path in paths if path.stem not in forbidden]
    if len(paths) > args.max_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_frames, dtype=int)]

    rng = np.random.default_rng(args.seed)
    clouds, counts = [], []
    for path in paths:
        points = load_xyz(path)
        counts.append(len(points))
        if len(points) > args.points_per_frame:
            points = points[rng.choice(len(points), args.points_per_frame, replace=False)]
        clouds.append(points)
    pooled = np.concatenate(clouds)
    radius = np.linalg.norm(pooled, axis=1)
    elevation = np.arctan2(pooled[:, 2], np.linalg.norm(pooled[:, :2], axis=1))
    azimuth = np.arctan2(pooled[:, 1], pooled[:, 0])
    ring_angles = fit_rings(elevation, args.rings)
    ring_id = np.abs(elevation[:, None] - ring_angles[None]).argmin(axis=1)

    ring_counts = np.bincount(ring_id, minlength=args.rings)
    ring_valid_rate = ring_counts / max(ring_counts.max(), 1)
    range_ring = np.zeros((args.rings, len(RANGE_EDGES) - 1), dtype=np.float64)
    azimuth_counts = np.zeros(len(AZIMUTH_EDGES) - 1, dtype=np.float64)
    for ring in range(args.rings):
        range_ring[ring], _ = np.histogram(radius[ring_id == ring], RANGE_EDGES)
    azimuth_counts, _ = np.histogram(azimuth, AZIMUTH_EDGES)
    range_ring /= np.maximum(range_ring.max(axis=1, keepdims=True), 1)
    azimuth_keep = azimuth_counts / max(azimuth_counts.max(), 1)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "ring_angles.npy", ring_angles)
    np.save(args.output / "ring_valid_rate.npy", ring_valid_rate.astype(np.float32))
    np.save(args.output / "range_ring_keep_prob.npy", range_ring.astype(np.float32))
    np.save(args.output / "azimuth_keep_prob.npy", azimuth_keep.astype(np.float32))
    np.savez(
        args.output / "multiscale_occupancy.npz",
        **{
            f"voxel_{size:g}": np.unique(np.floor(pooled / size).astype(np.int32), axis=0)
            for size in (0.25, 0.5, 1.0, 2.0)
        },
    )
    summary = {
        "policy": "unlabeled train -1 frames only; exclude every forbidden basename suffix",
        "frames": len(paths),
        "pooled_points": len(pooled),
        "rings": args.rings,
        "ring_angle_degrees": [float(x) for x in np.rad2deg(ring_angles)],
        "range_edges_m": RANGE_EDGES.tolist(),
        "azimuth_edges_rad": AZIMUTH_EDGES.tolist(),
        "point_count_distribution": {
            key: float(value)
            for key, value in zip(
                ("min", "p10", "median", "p90", "max"),
                np.percentile(counts, (0, 10, 50, 90, 100)),
            )
        },
        "source_paths": [str(path.relative_to(args.real_root)) for path in paths],
    }
    (args.output / "sensor_profile.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "source_paths"}, indent=2))


if __name__ == "__main__":
    main()
