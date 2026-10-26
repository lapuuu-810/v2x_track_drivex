#!/usr/bin/env python3
"""Distributionally robust, bounded sliced transport for synthetic LiDAR.

The transform is fitted to an equal-weight mixture of M1 and M2 *training*
scenarios.  It starts from a submitted synthetic anchor and only makes small
population-level displacements; no real scan can be copied to an output cloud.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_bounded_sensor_calibration import HIGH, LOW, forbidden_suffixes, load_anchor, load_real, transport_step, write_candidate


def scenarios(root: Path, agent_dir: str, forbidden: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.rglob(agent_dir) if path.is_dir()):
        files = [path for path in sorted(directory.glob("*.bin")) if path.stem not in forbidden]
        if files:
            result[str(directory.parent.relative_to(root))] = files
    if not result:
        raise RuntimeError(f"no allowed {agent_dir} scans under {root}")
    return result


def balanced_domain_target(groups: dict[str, list[Path]], count: int, frames_per_scenario: int, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    quota = int(np.ceil(count / len(groups)))
    pieces: list[np.ndarray] = []
    audit: list[dict[str, int | str]] = []
    for name, paths in sorted(groups.items()):
        take = np.linspace(0, len(paths) - 1, min(frames_per_scenario, len(paths)), dtype=int)
        clouds = [load_real(paths[i]) for i in take]
        clouds = [x for x in clouds if len(x)]
        if not clouds:
            continue
        pool = np.concatenate(clouds)
        pieces.append(pool[rng.choice(len(pool), quota, replace=len(pool) < quota)])
        audit.append({"scenario": name, "allowed_frames": len(paths), "fit_frames": len(clouds), "sampled_points": quota})
    pool = np.concatenate(pieces)
    return pool[rng.choice(len(pool), count, replace=len(pool) < count)].astype(np.float64), audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--m1-root", type=Path, required=True)
    p.add_argument("--m2-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--points-per-frame", type=int, default=9000)
    p.add_argument("--frames-per-scenario", type=int, default=18)
    p.add_argument("--rate", type=float, default=.025)
    p.add_argument("--snapshots", type=int, nargs="+", default=[0, 16, 32, 48, 64])
    p.add_argument("--seed", type=int, default=20260729)
    a = p.parse_args()
    if a.frames != 50 or a.points_per_frame * 12 > 1_500_000:
        raise ValueError("submission contract requires 50 files of <=1.5MB")
    if not 0 < a.rate <= .10:
        raise ValueError("rate must be in (0, .10]")
    snapshots = sorted(set(a.snapshots))
    if not snapshots or snapshots[0] != 0:
        raise ValueError("snapshots must include 0")
    bad = forbidden_suffixes(a.forbidden)
    m1 = scenarios(a.m1_root, "-1", bad)
    m2 = scenarios(a.m2_root, "-2", bad)
    points = load_anchor(a.anchor, a.frames, a.points_per_frame, a.seed).reshape(-1, 3)
    # Equal domain mass is the discrete Wasserstein-barycenter approximation.
    rng = np.random.default_rng(a.seed + 1)
    half = len(points) // 2
    target_m1, audit_m1 = balanced_domain_target(m1, half, a.frames_per_scenario, rng)
    target_m2, audit_m2 = balanced_domain_target(m2, len(points) - half, a.frames_per_scenario, rng)
    target = np.concatenate((target_m1, target_m2))
    rng = np.random.default_rng(a.seed + 2)
    captured = {0: points.copy()}
    for step in range(1, snapshots[-1] + 1):
        direction = np.eye(3)[step - 1] if step <= 3 else rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        transport_step(points, target, direction, a.rate)
        points[:] = np.clip(points, LOW, HIGH)
        if step in snapshots:
            captured[step] = points.copy()
    a.output_root.mkdir(parents=True, exist_ok=True)
    names = []
    for step in snapshots:
        name = "anchor" if step == 0 else f"dual_sot{step:02d}"
        write_candidate(a.output_root, name, captured[step], a.frames, a.points_per_frame)
        names.append(name)
    manifest = {
        "method": "equal-domain Wasserstein-barycenter bounded sliced sensor calibration",
        "anchor": str(a.anchor), "frames": a.frames, "points_per_frame": a.points_per_frame,
        "rate": a.rate, "snapshots": snapshots, "seed": a.seed,
        "policy": "outputs begin as synthetic anchor; public M1/M2 train scans only estimate a population transform; forbidden suffixes excluded; no validation/reference scans read",
        "m1_scenarios": audit_m1, "m2_scenarios": audit_m2, "candidates": names,
    }
    (a.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"m1_scenarios": len(audit_m1), "m2_scenarios": len(audit_m2), "candidates": names}, indent=2))


if __name__ == "__main__":
    main()
