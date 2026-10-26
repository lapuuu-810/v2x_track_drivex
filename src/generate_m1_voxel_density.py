#!/usr/bin/env python3
"""Sample new native-M1 point clouds from a training-split voxel density.

The density is fitted exclusively from corpus/<fit-split>.  Each source frame
contributes equal total mass, then all exported points are independently sampled
with within-voxel jitter.  Thus the output is a new distributional generator,
not a copy, blend, or transformation of validation/competition scans.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from export_lidm_reconstruction import native_range_to_xyz, PC_MIN, PC_MAX  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-split", choices=("train", "all_allowed"), default="train")
    parser.add_argument("--frames", type=int, default=142)
    parser.add_argument("--voxel-step", type=float, nargs=3, default=(0.5, 0.5, 0.2))
    parser.add_argument("--points-per-frame", type=int, default=0, help="0: resample training point-count distribution")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.fit_split == "all_allowed":
        # Final refit is permitted only because this manifest was built after
        # excluding every forbidden frame ID.  It never includes test inputs.
        manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
        if "exclude" not in manifest.get("forbidden_policy", "").lower():
            raise SystemExit("corpus manifest does not prove forbidden-frame exclusion")
        paths = sorted(args.corpus / row["subset"] / row["output"] for row in manifest["records"])
    else:
        paths = sorted((args.corpus / "train").glob("*.npz"))
    if not paths:
        raise SystemExit(f"no M1 scans for fitting in {args.corpus}")
    step = np.asarray(args.voxel_step, dtype=np.float64)
    span = PC_MAX.astype(np.float64) - PC_MIN.astype(np.float64)
    bins = np.rint(span / step).astype(np.int64)
    if not np.allclose(bins * step, span):
        raise SystemExit(f"voxel step {step.tolist()} does not divide cloud range {span.tolist()}")
    flat_bins = int(np.prod(bins))
    density = np.zeros(flat_bins, dtype=np.float64)
    ring_angles = np.load(args.corpus / "ring_angles.npy").astype(np.float32)
    source_counts = []
    for index, path in enumerate(paths, 1):
        with np.load(path) as item:
            xyz = native_range_to_xyz(item["log_range"], item["mask"].astype(bool), ring_angles)
        if len(xyz):
            grid = np.floor((xyz - PC_MIN) / step).astype(np.int64)
            grid = np.clip(grid, 0, bins - 1)
            flat = np.ravel_multi_index(grid.T, bins)
            # Equal mass per frame prevents long/dense scans dominating the fit.
            np.add.at(density, flat, 1.0 / len(xyz))
            source_counts.append(len(xyz))
        if index % 25 == 0 or index == len(paths):
            print(f"fit {index}/{len(paths)}", flush=True)
    if not source_counts:
        raise SystemExit("no valid points in fitting split")
    density /= density.sum()
    nonzero = np.flatnonzero(density)
    probabilities = density[nonzero]
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(args.seed)
    synthetic = args.output / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    counts = []
    for frame in range(args.frames):
        n = args.points_per_frame or int(source_counts[rng.integers(len(source_counts))])
        flat = rng.choice(nonzero, size=n, replace=True, p=probabilities)
        grid = np.column_stack(np.unravel_index(flat, bins))
        xyz = PC_MIN + (grid + rng.random((n, 3))) * step
        xyz = xyz.astype(np.float32)
        xyz.tofile(synthetic / f"{frame + 1:04d}.bin")
        counts.append(n)
    payload = {
        "method": "training-split frame-balanced voxel density with independent within-voxel sampling",
        "fit_split": args.fit_split,
        "fit_frames": len(paths),
        "validation_or_competition_frames_used": False,
        "voxel_step": step.tolist(), "bins": bins.tolist(),
        "frames": args.frames, "points": int(sum(counts)),
        "min_points": int(min(counts)), "max_points": int(max(counts)),
        "seed": args.seed,
    }
    (args.output / "generation_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
