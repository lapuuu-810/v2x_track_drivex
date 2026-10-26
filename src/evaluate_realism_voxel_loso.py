#!/usr/bin/env python3
"""LOSO evaluation of genuinely generated, smoothed voxel-density clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"),
]

import metrics  # noqa: E402
from evaluate_realism_baselines import normalized_score  # noqa: E402
from evaluate_realism_loso import collect_scenarios, fit_transforms  # noqa: E402
from generate_realism_baselines import PC_RANGE, load_xyz, pooled_points  # noqa: E402


def voxel_generate(
    points: np.ndarray,
    count: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    steps = np.asarray([0.5, 0.5, 0.2], dtype=np.float64)
    edges = [
        np.linspace(PC_RANGE[axis], PC_RANGE[axis + 3], int(round(
            (PC_RANGE[axis + 3] - PC_RANGE[axis]) / steps[axis]
        )) + 1)
        for axis in range(3)
    ]
    histogram, _ = np.histogramdd(points, bins=edges)
    if sigma > 0:
        histogram = gaussian_filter(histogram, sigma=sigma, mode="constant")
    probabilities = histogram.ravel()
    probabilities /= probabilities.sum()
    flat = rng.choice(len(probabilities), count, replace=True, p=probabilities)
    indices = np.column_stack(np.unravel_index(flat, histogram.shape))
    generated = np.empty((count, 3), dtype=np.float32)
    for axis in range(3):
        lower = np.asarray(edges[axis])[indices[:, axis]]
        generated[:, axis] = lower + rng.random(count) * steps[axis]
    return generated


def mix(first: np.ndarray, second: np.ndarray, second_fraction: float) -> np.ndarray:
    split = int(round(len(first) * (1.0 - second_fraction)))
    return np.concatenate([first[:split], second[:len(first) - split]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--generated-points", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    scenarios = collect_scenarios(args.real_root, args.forbidden)
    synthetic_paths = sorted(args.synthetic_root.glob("*.npy")) + sorted(args.synthetic_root.glob("*.bin"))
    synthetic = pooled_points(synthetic_paths, 320, 10_000, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    base = synthetic[rng.choice(len(synthetic), args.generated_points, replace=False)]
    metrics.N_SUBSAMPLE = args.subsample
    folds = []

    for fold, (held_name, held_paths) in enumerate(scenarios.items()):
        pools = []
        for offset, (name, paths) in enumerate(scenarios.items()):
            if name != held_name:
                pools.append(pooled_points(paths, 80, 8000, args.seed + fold * 1000 + offset))
        equal_count = min(map(len, pools))
        fold_rng = np.random.default_rng(args.seed + 10_000 + fold)
        real_fit = np.concatenate([
            points[fold_rng.choice(len(points), equal_count, replace=False)] for points in pools
        ])
        transformed = fit_transforms(base, synthetic, real_fit)["quantile_0p75"]
        voxel = {
            sigma: voxel_generate(real_fit, args.generated_points, sigma, fold_rng)
            for sigma in (0.0, 0.75, 1.5)
        }
        candidates = {
            "synthetic_raw": base,
            "quantile_0p75": transformed,
            "voxel_s0": voxel[0.0],
            "voxel_s0p75": voxel[0.75],
            "voxel_s1p5": voxel[1.5],
            "raw_voxel25": mix(base, voxel[0.75], 0.25),
            "raw_voxel50": mix(base, voxel[0.75], 0.50),
            "raw_voxel75": mix(base, voxel[0.75], 0.75),
            "quantile_voxel25": mix(transformed, voxel[0.75], 0.25),
            "quantile_voxel50": mix(transformed, voxel[0.75], 0.50),
            "quantile_voxel75": mix(transformed, voxel[0.75], 0.75),
        }
        if len(held_paths) > 80:
            indices = np.linspace(0, len(held_paths) - 1, 80, dtype=int)
            held_paths = [held_paths[index] for index in indices]
        validation = [load_xyz(path) for path in held_paths]
        current = {"held_scenario": held_name, "results": {}}
        for name, generated in candidates.items():
            generated = np.clip(generated, PC_RANGE[:3], PC_RANGE[3:]).astype(np.float32)
            result = metrics.compute_realism(
                [generated], validation, tuple(PC_RANGE.tolist()), seed=1234
            )
            result["normalized_realism"] = normalized_score(result)
            current["results"][name] = result
            print(
                f"fold={fold + 1}/{len(scenarios)} held={held_name} {name}: "
                f"{result['normalized_realism']:.6f}", flush=True
            )
        folds.append(current)

    summary = {}
    for name in folds[0]["results"]:
        values = np.asarray([fold["results"][name]["normalized_realism"] for fold in folds])
        summary[name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "folds": values.tolist(),
        }
    payload = {
        "protocol": "leave-one-scenario-out, equal fit-scenario weights",
        "voxel_steps": [0.5, 0.5, 0.2],
        "subsample": args.subsample,
        "summary": summary,
        "fold_results": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY")
    for name, result in sorted(summary.items(), key=lambda item: item[1]["mean"], reverse=True):
        print(
            f"{name:20s} mean={result['mean']:.6f} std={result['std']:.6f} "
            f"min={result['min']:.6f}", flush=True
        )


if __name__ == "__main__":
    main()
