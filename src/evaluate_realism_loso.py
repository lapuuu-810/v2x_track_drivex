#!/usr/bin/env python3
"""Leave-one-scenario-out evaluation for robust realism transformations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"),
]

import metrics  # noqa: E402
from evaluate_realism_baselines import normalized_score  # noqa: E402
from generate_realism_baselines import PC_RANGE, load_xyz, matrix_sqrt, pooled_points, quantile_map  # noqa: E402


def collect_scenarios(root: Path, forbidden_path: Path) -> dict[str, list[Path]]:
    forbidden_suffixes = {
        line.strip().split("_")[-1]
        for line in forbidden_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    output = {}
    for scenario in sorted(path for path in root.iterdir() if path.is_dir()):
        infra = scenario / "-1"
        paths = [
            path for path in sorted(infra.glob("*.bin"))
            if path.stem not in forbidden_suffixes
        ] if infra.is_dir() else []
        if paths:
            output[scenario.name] = paths
    return output


def scenario_pool(paths: list[Path], max_frames: int, points_per_frame: int, seed: int) -> np.ndarray:
    return pooled_points(paths, max_frames, points_per_frame, seed)


def fit_transforms(base: np.ndarray, synthetic: np.ndarray, real: np.ndarray) -> dict[str, np.ndarray]:
    syn_mean, real_mean = synthetic.mean(axis=0), real.mean(axis=0)
    syn_std, real_std = synthetic.std(axis=0), real.std(axis=0)
    axis = (base - syn_mean) * (real_std / np.maximum(syn_std, 1e-6)) + real_mean

    syn_cov, real_cov = np.cov(synthetic, rowvar=False), np.cov(real, rowvar=False)
    color = matrix_sqrt(syn_cov, inverse=True) @ matrix_sqrt(real_cov)
    covariance = (base - syn_mean) @ color + real_mean
    quantile = quantile_map(base, synthetic, real)

    candidates = {"synthetic_raw": base}
    for amount in (0.25, 0.5, 0.75, 1.0):
        suffix = str(amount).replace(".", "p")
        candidates[f"axis_{suffix}"] = base + amount * (axis - base)
        candidates[f"cov_{suffix}"] = base + amount * (covariance - base)
        candidates[f"quantile_{suffix}"] = base + amount * (quantile - base)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=1500)
    parser.add_argument("--max-fit-frames-per-scenario", type=int, default=80)
    parser.add_argument("--max-validation-frames", type=int, default=80)
    parser.add_argument("--points-per-source-frame", type=int, default=8000)
    parser.add_argument("--generated-points", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    scenarios = collect_scenarios(args.real_root, args.forbidden)
    if len(scenarios) < 2:
        raise SystemExit(f"Need at least two scenarios, found {len(scenarios)}")
    synthetic_paths = sorted(args.synthetic_root.glob("*.npy")) + sorted(args.synthetic_root.glob("*.bin"))
    synthetic = pooled_points(synthetic_paths, 320, 10_000, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    base = synthetic[rng.choice(len(synthetic), args.generated_points, replace=False)]
    metrics.N_SUBSAMPLE = args.subsample

    fold_results = []
    for fold, (held_name, held_paths) in enumerate(scenarios.items()):
        fit_by_scenario = []
        for offset, (name, paths) in enumerate(scenarios.items()):
            if name == held_name:
                continue
            fit_by_scenario.append(
                scenario_pool(
                    paths,
                    args.max_fit_frames_per_scenario,
                    args.points_per_source_frame,
                    args.seed + 1000 * fold + offset,
                )
            )
        # Equalize contribution across fit scenarios to prevent long sequences
        # from dominating the aggregate target distribution.
        equal_count = min(len(points) for points in fit_by_scenario)
        fit_parts = []
        fold_rng = np.random.default_rng(args.seed + 10_000 + fold)
        for points in fit_by_scenario:
            fit_parts.append(points[fold_rng.choice(len(points), equal_count, replace=False)])
        real_fit = np.concatenate(fit_parts)

        candidates = fit_transforms(base, synthetic, real_fit)
        empirical = real_fit[fold_rng.choice(len(real_fit), args.generated_points, replace=False)].copy()
        noise = fold_rng.normal(size=empirical.shape).astype(np.float32)
        for bandwidth in (0.02, 0.08, 0.20):
            scale = np.asarray([bandwidth, bandwidth, bandwidth * 0.5], dtype=np.float32)
            candidates[f"empirical_{bandwidth:.2f}"] = empirical + noise * scale
        mean, covariance = real_fit.mean(axis=0), np.cov(real_fit, rowvar=False)
        candidates["gaussian"] = fold_rng.multivariate_normal(
            mean, covariance, size=args.generated_points
        ).astype(np.float32)

        if len(held_paths) > args.max_validation_frames:
            indices = np.linspace(0, len(held_paths) - 1, args.max_validation_frames, dtype=int)
            held_paths = [held_paths[index] for index in indices]
        validation = [load_xyz(path) for path in held_paths]
        current = {
            "held_scenario": held_name,
            "fit_scenarios": len(fit_by_scenario),
            "fit_points": len(real_fit),
            "validation_frames": len(validation),
            "results": {},
        }
        for name, generated in candidates.items():
            generated = np.clip(generated, PC_RANGE[:3], PC_RANGE[3:]).astype(np.float32)
            result = metrics.compute_realism(
                [generated], validation, tuple(PC_RANGE.tolist()), seed=1234
            )
            result["normalized_realism"] = normalized_score(result)
            current["results"][name] = result
            print(
                f"fold={fold + 1}/{len(scenarios)} held={held_name} {name}: "
                f"{result['normalized_realism']:.6f}",
                flush=True,
            )
        fold_results.append(current)

    names = list(fold_results[0]["results"])
    summary = {}
    for name in names:
        values = np.asarray([
            fold["results"][name]["normalized_realism"] for fold in fold_results
        ])
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "folds": values.tolist(),
        }
    payload = {
        "protocol": "leave-one-scenario-out, equal fit-scenario weighting",
        "subsample": args.subsample,
        "scenario_count": len(scenarios),
        "summary": summary,
        "fold_results": fold_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY")
    for name, result in sorted(summary.items(), key=lambda item: item[1]["mean"], reverse=True):
        print(
            f"{name:20s} mean={result['mean']:.6f} std={result['std']:.6f} "
            f"min={result['min']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
