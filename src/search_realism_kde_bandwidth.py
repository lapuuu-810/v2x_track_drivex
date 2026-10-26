#!/usr/bin/env python3
"""Five-fold bandwidth search for the aggregate XYZ KDE realism model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORING_ROOT = PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(SCORING_ROOT)]

import metrics  # noqa: E402
from evaluate_realism_baselines import normalized_score  # noqa: E402
from generate_realism_baselines import (  # noqa: E402
    PC_RANGE,
    collect_real_frames,
    load_xyz,
    pooled_points,
    stable_bucket,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fit-frames", type=int, default=240)
    parser.add_argument("--validation-frames", type=int, default=60)
    parser.add_argument("--generated-points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    # collect_real_frames already applies the conservative forbidden-suffix filter;
    # combine its deterministic split and repartition for cross-validation.
    first, second = collect_real_frames(args.real_root, args.forbidden)
    paths = sorted(first + second)
    bandwidths = [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.30]
    metrics.N_SUBSAMPLE = args.subsample
    fold_results = []
    for fold in range(args.folds):
        validation_paths = [
            path for path in paths
            if stable_bucket(str(path.relative_to(args.real_root)), args.folds) == fold
        ]
        fit_paths = [path for path in paths if path not in set(validation_paths)]
        fit = pooled_points(fit_paths, args.fit_frames, 10000, args.seed + fold)
        if len(validation_paths) > args.validation_frames:
            indices = np.linspace(0, len(validation_paths) - 1, args.validation_frames, dtype=int)
            validation_paths = [validation_paths[index] for index in indices]
        validation = [load_xyz(path) for path in validation_paths]
        rng = np.random.default_rng(args.seed + 100 + fold)
        centers = fit[rng.choice(len(fit), args.generated_points, replace=False)]
        current = {"fold": fold, "fit_frames": len(fit_paths), "validation_frames": len(validation)}
        for bandwidth in bandwidths:
            noise = rng.normal(size=centers.shape).astype(np.float32)
            scale = np.asarray([bandwidth, bandwidth, bandwidth * 0.5], dtype=np.float32)
            generated = np.clip(centers + noise * scale, PC_RANGE[:3], PC_RANGE[3:])
            result = metrics.compute_realism(
                [generated.astype(np.float32)], validation, tuple(PC_RANGE.tolist()), seed=1234
            )
            result["normalized_realism"] = normalized_score(result)
            current[str(bandwidth)] = result
            print(f"fold={fold} bandwidth={bandwidth}: {json.dumps(result)}", flush=True)
        fold_results.append(current)

    summary = {}
    for bandwidth in bandwidths:
        key = str(bandwidth)
        values = np.asarray([fold[key]["normalized_realism"] for fold in fold_results])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "folds": values.tolist(),
        }
    payload = {"subsample": args.subsample, "summary": summary, "fold_results": fold_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
