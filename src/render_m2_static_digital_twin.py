#!/usr/bin/env python3
"""Render novel LiDAR scans from allowed V2X-Real M2 static digital twins.

This is deliberately different from a point-cloud copy or a distributional
transport.  For each fixed infrastructure sensor scene, a temporal-stable
voxel map is reconstructed from multiple *allowed* train scans.  A new sensor
pose is sampled near the original pose and the voxel map is ray-rendered from
that pose.  Consequently every emitted point is a voxel-surface intersection
under a new pose; no raw real scan is emitted or re-used as an output frame.

The output is intended as a LiDAR-RT-style physical baseline.  It uses only
``train_lidar128_agent_m2/**/-2/*.bin`` files whose frame suffix is absent
from ``forbidden_frames.txt``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def crop(points: np.ndarray) -> np.ndarray:
    keep = np.isfinite(points).all(axis=1) & np.any(points != 0.0, axis=1)
    keep &= (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
    return np.asarray(points[keep], dtype=np.float32)


def forbidden_suffixes(path: Path) -> set[str]:
    return {
        line.strip().split("_")[-1]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def discover(root: Path, forbidden: set[str]) -> dict[str, list[Path]]:
    scenes: dict[str, list[Path]] = {}
    for sensor_dir in sorted(p for p in root.rglob("-2") if p.is_dir()):
        paths = [p for p in sorted(sensor_dir.glob("*.bin")) if p.stem not in forbidden]
        if paths:
            scenes[str(sensor_dir.parent.relative_to(root))] = paths
    if not scenes:
        raise RuntimeError("no allowed M2 -2 scans found")
    return scenes


def choose_evenly(paths: list[Path], n: int) -> list[Path]:
    indices = np.linspace(0, len(paths) - 1, min(n, len(paths)), dtype=np.int64)
    return [paths[int(i)] for i in indices]


def stable_voxels(paths: list[Path], voxel_m: float, min_hits: int) -> np.ndarray:
    counts: Counter[tuple[int, int, int]] = Counter()
    for path in paths:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"expected XYZI real scan: {path}")
        xyz = crop(raw.reshape(-1, 4)[:, :3])
        codes = np.floor((xyz - LOW) / voxel_m).astype(np.int16)
        counts.update(map(tuple, np.unique(codes, axis=0)))
    codes = np.asarray([k for k, v in counts.items() if v >= min_hits], dtype=np.int16)
    if len(codes) == 0:
        raise RuntimeError("empty stable map; lower --min-hits")
    return LOW + (codes.astype(np.float32) + 0.5) * voxel_m


def render(
    world: np.ndarray,
    dx: float,
    dy: float,
    yaw: float,
    elevation_bins: int,
    azimuth_bins: int,
    output_points: int,
    noise_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return closest voxel-surface point per ray from a perturbed pose."""
    c, s = np.cos(yaw), np.sin(yaw)
    q = world.copy()
    q[:, 0] -= dx
    q[:, 1] -= dy
    # world -> virtual sensor coordinates, inverse yaw rotation
    x, y = q[:, 0].copy(), q[:, 1].copy()
    q[:, 0] = c * x + s * y
    q[:, 1] = -s * x + c * y
    q = crop(q)
    radial_xy = np.linalg.norm(q[:, :2], axis=1)
    radius = np.linalg.norm(q, axis=1)
    elevation = np.arctan2(q[:, 2], np.maximum(radial_xy, 1e-6))
    azimuth = np.arctan2(q[:, 1], q[:, 0])
    ie = np.clip(((elevation + 0.35) / 0.70 * elevation_bins).astype(np.int32), 0, elevation_bins - 1)
    ia = np.clip(((azimuth + np.pi) / (2.0 * np.pi) * azimuth_bins).astype(np.int32), 0, azimuth_bins - 1)
    flat = ie.astype(np.int64) * azimuth_bins + ia
    order = np.argsort(radius, kind="stable")
    # First index for each angular cell after sorting nearest to farthest.
    _, first = np.unique(flat[order], return_index=True)
    points = q[order[first]]
    if noise_m > 0:
        unit = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-6)
        points = points + unit * rng.normal(0.0, noise_m, size=(len(points), 1)).astype(np.float32)
    if len(points) < output_points:
        # Surface cells may be sparse after a large virtual displacement.
        # Sampling with replacement is still a fresh ray render, but is
        # reported explicitly so such a setup can be rejected in validation.
        chosen = rng.choice(len(points), output_points, replace=True)
    else:
        chosen = rng.choice(len(points), output_points, replace=False)
    return np.clip(points[chosen], LOW, HIGH).astype(np.float32), len(points) / max(elevation_bins * azimuth_bins, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-m2-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--maps", type=int, default=25, help="number of distinct train scenes to reconstruct")
    p.add_argument("--frames-per-map", type=int, default=24)
    p.add_argument("--voxel-m", type=float, default=0.35)
    p.add_argument("--min-hit-fraction", type=float, default=0.15)
    p.add_argument("--output-points", type=int, default=9000)
    p.add_argument("--elevation-bins", type=int, default=128)
    p.add_argument("--azimuth-bins", type=int, default=2048)
    p.add_argument("--translation-std-m", type=float, default=0.20)
    p.add_argument("--translation-max-m", type=float, default=0.50)
    p.add_argument("--yaw-std-deg", type=float, default=0.20)
    p.add_argument("--yaw-max-deg", type=float, default=0.60)
    p.add_argument("--radial-noise-m", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20260802)
    a = p.parse_args()
    if a.frames <= 0 or a.maps <= 0 or a.output_points <= 0:
        raise ValueError("frames, maps, and output-points must be positive")
    forbidden = forbidden_suffixes(a.forbidden)
    scenes = discover(a.real_m2_root, forbidden)
    # Deterministic evenly spaced scene selection avoids one long sequence
    # dominating the generated dataset-level point distribution.
    scene_names = sorted(scenes)
    scene_names = [scene_names[i] for i in np.linspace(0, len(scene_names) - 1, min(a.maps, len(scene_names)), dtype=np.int64)]
    rng = np.random.default_rng(a.seed)
    maps: list[tuple[str, np.ndarray, int]] = []
    for index, name in enumerate(scene_names, 1):
        selected = choose_evenly(scenes[name], a.frames_per_map)
        min_hits = max(2, int(np.ceil(len(selected) * a.min_hit_fraction)))
        world = stable_voxels(selected, a.voxel_m, min_hits)
        maps.append((name, world, len(selected)))
        print(f"map {index}/{len(scene_names)}: {name}, voxels={len(world)}", flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    for i in range(a.frames):
        name, world, used = maps[i % len(maps)]
        dx, dy = np.clip(rng.normal(0.0, a.translation_std_m, size=2), -a.translation_max_m, a.translation_max_m)
        yaw = float(np.clip(rng.normal(0.0, a.yaw_std_deg), -a.yaw_max_deg, a.yaw_max_deg) * np.pi / 180.0)
        points, coverage = render(world, float(dx), float(dy), yaw, a.elevation_bins, a.azimuth_bins, a.output_points, a.radial_noise_m, rng)
        points.tofile(a.output / f"{i + 1:04d}.bin")
        report.append({"frame": f"{i + 1:04d}.bin", "scene": name, "map_frames": used, "map_voxels": int(len(world)), "dx": float(dx), "dy": float(dy), "yaw_rad": yaw, "surface_cell_coverage": float(coverage), "points": int(len(points))})
    manifest = {
        "method": "per-scene temporal-stable voxel digital twin + perturbed-pose closest-surface ray rendering",
        "real_data_policy": "allowed train_lidar128_agent_m2/-2 only; forbidden suffixes excluded; outputs are voxel-surface re-renders, never raw real scan copies",
        "source_scenes": len(scene_names), "source_scans": int(sum(x[2] for x in maps)), "frames": report,
        "voxel_m": a.voxel_m, "min_hit_fraction": a.min_hit_fraction,
        "virtual_pose": {"translation_std_m": a.translation_std_m, "translation_max_m": a.translation_max_m, "yaw_std_deg": a.yaw_std_deg, "yaw_max_deg": a.yaw_max_deg},
        "output_points": a.output_points, "radial_noise_m": a.radial_noise_m, "seed": a.seed,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
