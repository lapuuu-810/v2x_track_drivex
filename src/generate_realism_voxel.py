#!/usr/bin/env python3
"""Generate synthetic XYZ frames from a scenario-balanced voxel density."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


PC_RANGE = np.asarray([-40.0, -40.0, -8.0, 40.0, 40.0, 2.0], dtype=np.float64)


def load_xyz4(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Expected four-column V2X-Real LiDAR file: {path}")
    points = raw.reshape(-1, 4)[:, :3]
    mask = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    mask &= (points >= PC_RANGE[:3]).all(axis=1) & (points <= PC_RANGE[3:]).all(axis=1)
    return points[mask]


def discover_scenarios(roots: list[Path]) -> list[Path]:
    scenarios = set()
    for root in roots:
        for infra in root.rglob("-1"):
            if infra.is_dir() and any(infra.glob("*.bin")):
                scenarios.add(infra.parent.resolve())
    return sorted(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, action="append", required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--voxel-step", type=float, nargs=3, default=(0.5, 0.5, 0.2))
    parser.add_argument("--smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--max-frames-per-scenario", type=int, default=160)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    forbidden_suffixes = {
        line.strip().split("_")[-1]
        for line in args.forbidden.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    step = np.asarray(args.voxel_step, dtype=np.float64)
    spans = PC_RANGE[3:] - PC_RANGE[:3]
    bins = np.rint(spans / step).astype(int)
    if not np.allclose(bins * step, spans):
        raise SystemExit("voxel-step must evenly divide the point-cloud range")
    edges = [
        np.linspace(PC_RANGE[axis], PC_RANGE[axis + 3], bins[axis] + 1)
        for axis in range(3)
    ]

    scenarios = discover_scenarios(args.real_root)
    if not scenarios:
        raise SystemExit("No scenario containing an infrastructure '-1' directory was found")
    density = np.zeros(tuple(bins), dtype=np.float64)
    scenario_records = []
    total_forbidden = 0
    for scenario_index, scenario in enumerate(scenarios, 1):
        all_paths = sorted((scenario / "-1").glob("*.bin"))
        allowed = [path for path in all_paths if path.stem not in forbidden_suffixes]
        total_forbidden += len(all_paths) - len(allowed)
        if len(allowed) > args.max_frames_per_scenario:
            indices = np.linspace(0, len(allowed) - 1, args.max_frames_per_scenario, dtype=int)
            selected = [allowed[index] for index in indices]
        else:
            selected = allowed
        scenario_hist = np.zeros_like(density)
        point_count = 0
        for path in selected:
            points = load_xyz4(path)
            current, _ = np.histogramdd(points, bins=edges)
            scenario_hist += current
            point_count += len(points)
        if point_count == 0:
            continue
        scenario_hist /= scenario_hist.sum()
        density += scenario_hist
        scenario_records.append({
            "scenario": scenario.name,
            "source_root": next(
                str(root) for root in args.real_root
                if scenario == root.resolve() or root.resolve() in scenario.parents
            ),
            "available_frames": len(all_paths),
            "allowed_frames": len(allowed),
            "selected_frames": len(selected),
            "selected_points": point_count,
        })
        print(
            f"[{scenario_index}/{len(scenarios)}] {scenario.name}: "
            f"{len(selected)} frames, {point_count:,} points",
            flush=True,
        )
    if not scenario_records:
        raise SystemExit("No usable scenario remained after filtering")
    density /= density.sum()
    if args.smoothing_sigma > 0:
        density = gaussian_filter(density, sigma=args.smoothing_sigma, mode="constant")
        density /= density.sum()

    rng = np.random.default_rng(args.seed)
    total = args.frames * args.points_per_frame
    flat = rng.choice(density.size, total, replace=True, p=density.ravel())
    indices = np.column_stack(np.unravel_index(flat, density.shape))
    generated = np.empty((total, 3), dtype=np.float32)
    for axis in range(3):
        lower = np.asarray(edges[axis])[indices[:, axis]]
        generated[:, axis] = lower + rng.random(total) * step[axis]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(
        generated.reshape(args.frames, args.points_per_frame, 3), 1
    ):
        frame.tofile(args.output_dir / f"{index:04d}.bin")
    payload = {
        "method": "scenario-balanced voxel density sampling",
        "real_roots": [str(path) for path in args.real_root],
        "forbidden_policy": "exclude every basename suffix occurring anywhere in forbidden_frames.txt",
        "forbidden_files_excluded": total_forbidden,
        "voxel_step": step.tolist(),
        "smoothing_sigma_bins": args.smoothing_sigma,
        "point_cloud_range": PC_RANGE.tolist(),
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "seed": args.seed,
        "scenario_count": len(scenario_records),
        "scenarios": scenario_records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
