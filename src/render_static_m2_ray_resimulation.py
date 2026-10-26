#!/usr/bin/env python3
"""LiDAR-RT-inspired static-background ray re-simulation for V2X-Real.

Allowed M2 training scans are fused only into *time-stable voxel surfaces*.
Each output retains a released UrbanTwin synthetic scan's ray directions and
dynamic support; where a stable background surface is visible, a newly rendered
range is produced from the fused surface plus continuous radial noise.  No real
scan or real point is copied to an output frame.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def crop(points: np.ndarray) -> np.ndarray:
    ok = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    ok &= (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
    return np.asarray(points[ok], dtype=np.float32)


def discover(root: Path, forbidden: set[str]) -> dict[str, list[Path]]:
    scenes: dict[str, list[Path]] = {}
    for sensor in sorted(path for path in root.rglob("-2") if path.is_dir()):
        paths = [path for path in sorted(sensor.glob("*.bin")) if path.stem not in forbidden]
        if paths:
            scenes[str(sensor.parent.relative_to(root))] = paths
    if not scenes:
        raise RuntimeError("no allowed M2 scans found")
    return scenes


def angular_indices(points: np.ndarray, elev_bins: int, azimuth_bins: int) -> tuple[np.ndarray, np.ndarray]:
    radius_xy = np.linalg.norm(points[:, :2], axis=1)
    elevation = np.arctan2(points[:, 2], np.maximum(radius_xy, 1e-6))
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    ie = np.clip(((elevation + 0.35) / 0.70 * elev_bins).astype(np.int32), 0, elev_bins - 1)
    ia = np.clip(((azimuth + np.pi) / (2 * np.pi) * azimuth_bins).astype(np.int32), 0, azimuth_bins - 1)
    return ie, ia


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--real-m2-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xy-scale", type=float, default=0.96)
    parser.add_argument("--z-scale", type=float, default=0.85)
    parser.add_argument("--voxel-m", type=float, default=0.25)
    parser.add_argument("--max-frames-per-scene", type=int, default=80)
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--elevation-bins", type=int, default=128)
    parser.add_argument("--azimuth-bins", type=int, default=2048)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--max-range-residual-m", type=float, default=12.0)
    parser.add_argument("--radial-noise-m", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if not 0 < args.blend <= 1:
        raise ValueError("blend must be in (0, 1]")
    forbidden = {line.strip().split("_")[-1] for line in args.forbidden.read_text().splitlines() if line.strip() and not line.startswith("#")}
    scenes = discover(args.real_m2_root, forbidden)
    # Each frame contributes a binary voxel set.  Temporal recurrence, rather
    # than raw point multiplicity, identifies the stationary background.
    counts: Counter[tuple[int, int, int]] = Counter()
    frames_used = 0
    for scene_index, paths in enumerate(scenes.values(), 1):
        selected = [paths[i] for i in np.linspace(0, len(paths) - 1, min(len(paths), args.max_frames_per_scene), dtype=int)]
        for path in selected:
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size % 4:
                raise ValueError(f"unexpected real XYZI layout: {path}")
            xyz = crop(raw.reshape(-1, 4)[:, :3])
            codes = np.floor((xyz - LOW) / args.voxel_m).astype(np.int16)
            counts.update(map(tuple, np.unique(codes, axis=0)))
            frames_used += 1
        print(f"fused {scene_index}/{len(scenes)} scenes", flush=True)
    retained = np.asarray([code for code, count in counts.items() if count >= args.min_hits], dtype=np.int16)
    if len(retained) == 0:
        raise RuntimeError("no stable voxels retained; reduce --min-hits")
    centers = LOW + (retained.astype(np.float32) + 0.5) * args.voxel_m
    ranges = np.linalg.norm(centers, axis=1)
    ie, ia = angular_indices(centers, args.elevation_bins, args.azimuth_bins)
    # A ray renderer keeps the nearest stable surface for each angular cell.
    surface = np.full((args.elevation_bins, args.azimuth_bins), np.inf, dtype=np.float32)
    np.minimum.at(surface, (ie, ia), ranges)
    anchor_paths = sorted(args.anchor.glob("*.bin"))
    if len(anchor_paths) != 50:
        raise ValueError(f"anchor must have 50 scans, got {len(anchor_paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frame_report = []
    for output_index, path in enumerate(anchor_paths, 1):
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 3:
            raise ValueError(f"unexpected synthetic XYZ layout: {path}")
        xyz = crop(raw.reshape(-1, 3)).copy()
        xyz[:, :2] *= args.xy_scale
        xyz[:, 2] *= args.z_scale
        radii = np.linalg.norm(xyz, axis=1)
        direction = xyz / np.maximum(radii[:, None], 1e-6)
        re, ra = angular_indices(xyz, args.elevation_bins, args.azimuth_bins)
        static_ranges = surface[re, ra]
        valid = np.isfinite(static_ranges) & (np.abs(static_ranges - radii) <= args.max_range_residual_m)
        rendered_ranges = (1.0 - args.blend) * radii[valid] + args.blend * static_ranges[valid]
        rendered_ranges += rng.normal(0.0, args.radial_noise_m, size=rendered_ranges.shape)
        xyz[valid] = direction[valid] * rendered_ranges[:, None]
        xyz = np.clip(xyz, LOW, HIGH).astype(np.float32)
        xyz.tofile(args.output / f"{output_index:04d}.bin")
        frame_report.append({"source": path.name, "points": int(len(xyz)), "static_ray_fraction": float(valid.mean())})
    manifest = {
        "method": "M2 time-stable voxel surface fused then ray-consistently re-rendered along UrbanTwin synthetic rays",
        "real_data_policy": "allowed M2 train/-2 scans only; forbidden basename suffixes excluded; no real point copied to output",
        "anchor": str(args.anchor), "real_m2_root": str(args.real_m2_root),
        "physical_scale": {"xy": args.xy_scale, "z": args.z_scale},
        "voxel_m": args.voxel_m, "min_temporal_hits": args.min_hits,
        "scenes": len(scenes), "frames_used": frames_used, "stable_voxels": int(len(retained)),
        "ray_grid": [args.elevation_bins, args.azimuth_bins], "blend": args.blend,
        "max_range_residual_m": args.max_range_residual_m, "radial_noise_m": args.radial_noise_m,
        "seed": args.seed, "frames": frame_report,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
