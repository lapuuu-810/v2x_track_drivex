#!/usr/bin/env python3
"""Bounded sensor-domain calibration for released UrbanTwin LiDAR scans.

The generator always starts from the released *synthetic* UrbanTwin source.
Public V2X-Real M1 training scans are used only to estimate a population-level
sensor transform.  A small sliced-transport step size and saved early snapshots
keep the geometry and point-process characteristics of the digital-twin source
intact; this is deliberately not an empirical real-point generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def forbidden_suffixes(path: Path) -> set[str]:
    return {
        line.strip().split("_")[-1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def discover_by_scenario(root: Path, forbidden: set[str]) -> dict[str, list[Path]]:
    scenarios: dict[str, list[Path]] = {}
    for infra in sorted(path for path in root.rglob("-1") if path.is_dir()):
        paths = [path for path in sorted(infra.glob("*.bin")) if path.stem not in forbidden]
        if paths:
            scenarios[str(infra.parent.relative_to(root))] = paths
    if not scenarios:
        raise RuntimeError(f"no allowed M1 training scans below {root}")
    return scenarios


def load_real(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"unexpected V2X-Real layout: {path}")
    points = raw.reshape(-1, 4)[:, :3]
    valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    valid &= (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
    return np.asarray(points[valid], dtype=np.float32)


def build_balanced_target(scenarios: dict[str, list[Path]], target_count: int, frames_per_scenario: int, seed: int) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    audit: list[dict[str, int | str]] = []
    quota = int(np.ceil(target_count / len(scenarios)))
    for name, paths in sorted(scenarios.items()):
        indices = np.linspace(0, len(paths) - 1, min(frames_per_scenario, len(paths)), dtype=int)
        clouds = [load_real(paths[index]) for index in indices]
        clouds = [cloud for cloud in clouds if len(cloud)]
        if not clouds:
            continue
        pool = np.concatenate(clouds)
        selected = pool[rng.choice(len(pool), quota, replace=len(pool) < quota)]
        pieces.append(selected)
        audit.append({"scenario": name, "allowed_frames": len(paths), "fit_frames": len(clouds), "target_points": len(selected)})
    target = np.concatenate(pieces)
    return target[rng.choice(len(target), target_count, replace=False)].astype(np.float64), audit


def load_anchor(root: Path, frames: int, points_per_frame: int, seed: int) -> np.ndarray:
    paths = sorted(root.glob("*.bin"))
    if len(paths) < frames:
        raise RuntimeError(f"anchor needs {frames} bins, found {len(paths)}")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for path in [paths[index] for index in np.linspace(0, len(paths) - 1, frames, dtype=int)]:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 3:
            raise ValueError(f"unexpected synthetic XYZ layout: {path}")
        cloud = raw.reshape(-1, 3)
        valid = np.isfinite(cloud).all(axis=1) & (cloud >= LOW).all(axis=1) & (cloud <= HIGH).all(axis=1)
        cloud = cloud[valid]
        selected.append(cloud[rng.choice(len(cloud), points_per_frame, replace=len(cloud) < points_per_frame)])
    return np.stack(selected).astype(np.float64)


def transport_step(points: np.ndarray, target: np.ndarray, direction: np.ndarray, rate: float) -> None:
    source_projection = points @ direction
    target_projection = target @ direction
    source_order = np.argsort(source_projection)
    points[source_order] += (rate * (np.sort(target_projection) - source_projection[source_order]))[:, None] * direction


def write_candidate(root: Path, name: str, points: np.ndarray, frames: int, points_per_frame: int) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for index, cloud in enumerate(points.reshape(frames, points_per_frame, 3), 1):
        np.asarray(cloud, dtype=np.float32).tofile(directory / f"{index:04d}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--real-train-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=9000)
    parser.add_argument("--frames-per-scenario", type=int, default=24)
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.frames != 50 or args.points_per_frame * 12 > 1_500_000:
        raise ValueError("submission contract requires 50 files of <=1.5MB")
    if args.rate <= 0 or args.rate > 0.10:
        raise ValueError("rate must be in (0, 0.10] for bounded calibration")
    snapshots = sorted(set(args.snapshots))
    if not snapshots or snapshots[0] != 0:
        raise ValueError("snapshots must include the unmodified anchor (0)")
    forbidden = forbidden_suffixes(args.forbidden)
    scenarios = discover_by_scenario(args.real_train_root, forbidden)
    frame_array = load_anchor(args.anchor, args.frames, args.points_per_frame, args.seed)
    points = frame_array.reshape(-1, 3)
    target, audit = build_balanced_target(scenarios, len(points), args.frames_per_scenario, args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    captured: dict[int, np.ndarray] = {0: points.copy()}
    for iteration in range(1, snapshots[-1] + 1):
        # First axes make the small sensor-origin and height correction stable;
        # later random slices correct only low-order global discrepancies.
        direction = np.eye(3)[iteration - 1] if iteration <= 3 else rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        transport_step(points, target, direction, args.rate)
        points[:] = np.clip(points, LOW, HIGH)
        if iteration in snapshots:
            captured[iteration] = points.copy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    names = []
    for step in snapshots:
        name = "anchor_native" if step == 0 else f"calibrated_sot{step:02d}"
        write_candidate(args.output_root, name, captured[step], args.frames, args.points_per_frame)
        names.append(name)
    manifest = {
        "method": "bounded scenario-balanced sliced sensor calibration of released UrbanTwin native LiDAR",
        "anchor": str(args.anchor),
        "real_train_root": str(args.real_train_root),
        "forbidden_policy": "globally exclude every forbidden basename suffix before fitting",
        "validation_policy": "no V2X-Real validation frames are read by this generator",
        "source_geometry_policy": "every candidate begins as a released UrbanTwin synthetic frame; only bounded population-level SOT displacement is applied",
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "rate": args.rate,
        "snapshots": snapshots,
        "seed": args.seed,
        "source_scenarios": len(audit),
        "scenarios": audit,
        "candidates": names,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
