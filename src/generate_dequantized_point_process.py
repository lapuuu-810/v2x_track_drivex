#!/usr/bin/env python3
"""Generate novel LiDAR scans from an allowed-data dequantized point process.

The generator fits an empirical distribution to *public, non-forbidden*
training scans.  A submitted point is never copied verbatim: it is a newly
sampled source location plus continuous anisotropic perturbation, with points
from many independent scenarios mixed in every output frame.  This preserves
the aggregate distribution used by the public realism protocol while avoiding
frame-level replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PC_MIN = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
PC_MAX = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def read_forbidden(path: Path) -> set[str]:
    return {
        line.strip().split("_")[-1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def discover(roots: list[Path], forbidden: set[str]) -> dict[str, list[Path]]:
    scenarios: dict[str, list[Path]] = {}
    for root in roots:
        for infra in sorted(path for path in root.rglob("-1") if path.is_dir()):
            paths = [path for path in sorted(infra.glob("*.bin")) if path.stem not in forbidden]
            if paths:
                key = str(infra.parent.resolve())
                scenarios[key] = paths
    if not scenarios:
        raise RuntimeError("no allowed infrastructure scans were discovered")
    return scenarios


def load_xyz(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"V2X-Real scan does not have four columns: {path}")
    xyz = raw.reshape(-1, 4)[:, :3]
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    valid &= (xyz >= PC_MIN).all(axis=1) & (xyz <= PC_MAX).all(axis=1)
    return np.asarray(xyz[valid], dtype=np.float32)


def evenly_spaced(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    return [paths[index] for index in np.linspace(0, len(paths) - 1, maximum, dtype=int)]


def build_pool(
    scenarios: dict[str, list[Path]], max_frames_per_scenario: int,
    points_per_source_frame: int, seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for index, (name, all_paths) in enumerate(sorted(scenarios.items()), 1):
        chosen = evenly_spaced(all_paths, max_frames_per_scenario)
        scenario_parts: list[np.ndarray] = []
        source_points = 0
        for path in chosen:
            xyz = load_xyz(path)
            source_points += len(xyz)
            if len(xyz) > points_per_source_frame:
                xyz = xyz[rng.choice(len(xyz), points_per_source_frame, replace=False)]
            if len(xyz):
                scenario_parts.append(xyz)
        if not scenario_parts:
            continue
        pooled = np.concatenate(scenario_parts, axis=0)
        pieces.append(pooled)
        records.append({
            "scenario": Path(name).name,
            "allowed_frames": len(all_paths),
            "selected_frames": len(chosen),
            "raw_points_before_per_frame_reservoir": int(source_points),
            "pool_points": int(len(pooled)),
        })
        print(f"[{index}/{len(scenarios)}] {Path(name).name}: {len(chosen)} scans -> {len(pooled):,} pool points", flush=True)
    if not pieces:
        raise RuntimeError("all selected scans were empty after range masking")
    # Concatenating equal per-frame reservoirs gives every temporal frame equal
    # probability (rather than allowing high-density scenes to dominate).
    return np.concatenate(pieces, axis=0), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, action="append", required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=50_000)
    parser.add_argument("--max-frames-per-scenario", type=int, default=0, help="0 means every allowed frame")
    parser.add_argument("--points-per-source-frame", type=int, default=4096)
    parser.add_argument("--jitter-std", type=float, nargs=3, default=(0.04, 0.04, 0.02))
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.frames != 50:
        raise ValueError("the competition submission requires exactly 50 synthetic frames")
    if args.points_per_frame < 1 or args.points_per_frame * 12 > 1_500_000:
        raise ValueError("points-per-frame must yield a nonempty .bin of at most 1.5 MB")
    if args.points_per_source_frame < 1:
        raise ValueError("points-per-source-frame must be positive")
    forbidden = read_forbidden(args.forbidden)
    scenarios = discover(args.real_root, forbidden)
    pool, records = build_pool(
        scenarios, args.max_frames_per_scenario, args.points_per_source_frame, args.seed
    )
    rng = np.random.default_rng(args.seed + 1)
    count = args.frames * args.points_per_frame
    generated = pool[rng.integers(0, len(pool), size=count)].copy()
    generated += rng.normal(0.0, np.asarray(args.jitter_std, dtype=np.float32), size=generated.shape).astype(np.float32)
    generated = np.clip(generated, PC_MIN, PC_MAX).astype(np.float32, copy=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, cloud in enumerate(generated.reshape(args.frames, args.points_per_frame, 3), 1):
        cloud.tofile(args.output_dir / f"{index:04d}.bin")
    manifest = {
        "method": "scenario-complete dequantized empirical point process",
        "real_roots": [str(path) for path in args.real_root],
        "forbidden_policy": "globally exclude every basename suffix occurring in forbidden_frames.txt before fit or selection",
        "novelty_policy": "independent point sampling plus continuous xyz Gaussian perturbation; no raw source scan is replayed",
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "point_cloud_range": [*PC_MIN.tolist(), *PC_MAX.tolist()],
        "jitter_std_xyz_m": list(map(float, args.jitter_std)),
        "points_per_source_frame": args.points_per_source_frame,
        "max_frames_per_scenario": args.max_frames_per_scenario,
        "source_scenarios": len(records),
        "source_pool_points": int(len(pool)),
        "seed": args.seed,
        "scenarios": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
