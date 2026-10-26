#!/usr/bin/env python3
"""Generate structure-preserving synthetic frames via sliced distribution transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_realism_baselines import PC_RANGE, collect_real_frames, pooled_points


def load_synthetic_xyz(path: Path) -> np.ndarray:
    """Read competition synthetic frames, which are raw XYZ float32."""
    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 3:
            raise ValueError(f"{path}: synthetic frame is not float32 XYZ")
        points = raw.reshape(-1, 3)
    mask = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    mask &= (points >= PC_RANGE[:3]).all(axis=1) & (points <= PC_RANGE[3:]).all(axis=1)
    return np.asarray(points[mask], dtype=np.float32)


def select_structured_frames(paths: list[Path], frames: int, points_per_frame: int, seed: int) -> np.ndarray:
    indices = np.linspace(0, len(paths) - 1, frames, dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for index in indices:
        points = load_synthetic_xyz(paths[index])
        if len(points) == points_per_frame:
            chosen = points
        else:
            chosen = points[rng.choice(len(points), points_per_frame, replace=len(points) < points_per_frame)]
        selected.append(chosen)
    return np.stack(selected).astype(np.float32)


def write_frames(root: Path, name: str, frames: np.ndarray) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for index, points in enumerate(frames, 1):
        np.asarray(points, dtype=np.float32).tofile(directory / f"{index:04d}.bin")


def transport_step(points: np.ndarray, target: np.ndarray, direction: np.ndarray, rate: float) -> None:
    source_projection = points @ direction
    target_projection = target @ direction
    source_order = np.argsort(source_projection)
    target_sorted = np.sort(target_projection)
    displacement = target_sorted - source_projection[source_order]
    points[source_order] += (rate * displacement)[:, None] * direction[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    real_fit_paths, validation_paths = collect_real_frames(args.real_root, args.forbidden)
    synthetic_paths = sorted(args.synthetic_root.glob("*.npy")) + sorted(args.synthetic_root.glob("*.bin"))
    frames = select_structured_frames(
        synthetic_paths, args.frames, args.points_per_frame, args.seed
    )
    points = frames.reshape(-1, 3).astype(np.float64)
    real_fit = pooled_points(real_fit_paths, 320, 12000, args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    target = real_fit[rng.choice(len(real_fit), len(points), replace=False)].astype(np.float64)
    snapshots = {0: points.copy()}
    axis_directions = np.eye(3, dtype=np.float64)
    for iteration in range(1, args.iterations + 1):
        if iteration <= 3:
            direction = axis_directions[iteration - 1]
        else:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
        transport_step(points, target, direction, args.rate)
        points[:] = np.clip(points, PC_RANGE[:3], PC_RANGE[3:])
        if iteration in {8, 16, 32, 64, 128, args.iterations}:
            snapshots[iteration] = points.copy()

    args.output_root.mkdir(parents=True, exist_ok=True)
    names = []
    for iteration, snapshot in snapshots.items():
        name = "sot_raw_structured" if iteration == 0 else f"sot_{iteration}"
        write_frames(
            args.output_root,
            name,
            snapshot.reshape(args.frames, args.points_per_frame, 3),
        )
        names.append(name)
    manifest = {
        "candidates": names,
        "validation_paths": [str(path) for path in validation_paths],
        "real_fit_frames": len(real_fit_paths),
        "real_validation_frames": len(validation_paths),
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "iterations": args.iterations,
        "rate": args.rate,
        "seed": args.seed,
        "method": "sliced optimal transport applied to points from intact synthetic frames",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "validation_paths"}, indent=2))


if __name__ == "__main__":
    main()
