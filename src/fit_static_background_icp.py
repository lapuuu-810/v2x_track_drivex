#!/usr/bin/env python3
"""Fit a physical LiDAR pose residual from static background geometry.

This is deliberately not distribution transport.  It estimates only a rigid
XY yaw/translation residual after a fixed, previously calibrated sensor scale,
using coarse, repeated background geometry in allowed public M2 training scans.
The fitted transform is reported but never applied implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def crop(points: np.ndarray) -> np.ndarray:
    good = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    good &= (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
    # Tall moving actors are intentionally excluded from registration.
    good &= (points[:, 2] < 0.8)
    return np.asarray(points[good], dtype=np.float32)


def voxelize(points: np.ndarray, size: float) -> np.ndarray:
    codes = np.floor((points - LOW) / size).astype(np.int32)
    unique = np.unique(codes, axis=0)
    return LOW + (unique.astype(np.float32) + 0.5) * size


def load_native(root: Path, xy_scale: float, z_scale: float) -> np.ndarray:
    clouds = []
    for path in sorted(root.glob("*.bin")):
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 3:
            raise ValueError(f"unexpected synthetic XYZ: {path}")
        xyz = crop(raw.reshape(-1, 3))
        xyz[:, :2] *= xy_scale
        xyz[:, 2] *= z_scale
        clouds.append(xyz)
    if len(clouds) != 50:
        raise ValueError(f"expected 50 native frames, got {len(clouds)}")
    return np.concatenate(clouds)


def discover_m2(root: Path, forbidden: set[str]) -> list[Path]:
    paths = []
    for sensor in sorted(path for path in root.rglob("-2") if path.is_dir()):
        paths.extend(path for path in sorted(sensor.glob("*.bin")) if path.stem not in forbidden)
    if not paths:
        raise ValueError("no allowed M2 training scans found")
    return paths


def load_real_pool(paths: list[Path], max_frames: int, seed: int) -> np.ndarray:
    if len(paths) > max_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_frames, dtype=int)]
    rng = np.random.default_rng(seed)
    chunks = []
    for path in paths:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"unexpected real XYZI: {path}")
        xyz = crop(raw.reshape(-1, 4)[:, :3])
        if len(xyz) > 6000:
            xyz = xyz[rng.choice(len(xyz), 6000, replace=False)]
        chunks.append(xyz)
    return np.concatenate(chunks)


def fit_rigid_xy(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares yaw and translation aligning source to target matches."""
    a, b = source[:, :2], target[:, :2]
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = cb - ca @ rotation.T
    return rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real-m2-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xy-scale", type=float, default=0.96)
    parser.add_argument("--z-scale", type=float, default=0.85)
    parser.add_argument("--max-real-frames", type=int, default=640)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-correspondence", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    forbidden = {line.strip().split("_")[-1] for line in args.forbidden.read_text().splitlines() if line.strip() and not line.startswith("#")}
    source = voxelize(load_native(args.synthetic, args.xy_scale, args.z_scale), args.voxel_size)
    real_paths = discover_m2(args.real_m2_root, forbidden)
    target = voxelize(load_real_pool(real_paths, args.max_real_frames, args.seed), args.voxel_size)
    # Bounded size keeps ICP deterministic and prevents a dense road patch from
    # dominating the global structure fit.
    rng = np.random.default_rng(args.seed + 1)
    if len(source) > 120_000:
        source = source[rng.choice(len(source), 120_000, replace=False)]
    if len(target) > 180_000:
        target = target[rng.choice(len(target), 180_000, replace=False)]
    tree = cKDTree(target)
    transformed = source.copy()
    rotation_total = np.eye(2)
    translation_total = np.zeros(2)
    history = []
    for iteration in range(args.iterations):
        distances, indices = tree.query(transformed, k=1, workers=-1)
        mask = distances <= args.max_correspondence
        if mask.sum() < 1000:
            raise RuntimeError("too few static background correspondences")
        rotation, translation = fit_rigid_xy(transformed[mask], target[indices[mask]])
        transformed[:, :2] = transformed[:, :2] @ rotation.T + translation
        rotation_total = rotation @ rotation_total
        translation_total = translation_total @ rotation.T + translation
        yaw = float(np.arctan2(rotation_total[1, 0], rotation_total[0, 0]))
        history.append({"iteration": iteration + 1, "matches": int(mask.sum()), "mean_nn_m": float(distances[mask].mean()), "median_nn_m": float(np.median(distances[mask])), "yaw_rad": yaw, "translation_xy_m": translation_total.tolist()})
    report = {
        "method": "static-background coarse-voxel ICP after fixed physical sensor scale",
        "synthetic": str(args.synthetic), "real_m2_root": str(args.real_m2_root),
        "forbidden_policy": "every forbidden basename suffix excluded before registration",
        "real_training_scans_available": len(real_paths), "real_training_scans_used": min(len(real_paths), args.max_real_frames),
        "fixed_sensor_scale": {"xy": args.xy_scale, "z": args.z_scale},
        "voxel_size_m": args.voxel_size, "max_correspondence_m": args.max_correspondence,
        "final_yaw_rad": float(np.arctan2(rotation_total[1, 0], rotation_total[0, 0])),
        "final_translation_xy_m": translation_total.tolist(),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
