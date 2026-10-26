#!/usr/bin/env python3
"""Evaluate generated realism candidates with the official metric implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORING_ROOT = PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"
sys.path.insert(0, str(SCORING_ROOT))

import metrics  # noqa: E402


PC_RANGE = (-40, -40, -8, 40, 40, 2)


def load_candidate(directory: Path) -> list[np.ndarray]:
    return [np.fromfile(path, np.float32).reshape(-1, 3) for path in sorted(directory.glob("*.bin"))]


def load_real(paths: list[str], max_frames: int) -> list[np.ndarray]:
    if len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[index] for index in indices]
    clouds = []
    for value in paths:
        raw = np.fromfile(value, np.float32)
        points = raw.reshape(-1, 4)[:, :3]
        points = points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]
        clouds.append(points.astype(np.float32, copy=False))
    return clouds


def normalized_score(result: dict) -> float:
    ranges = {"CD": (2.0, 30.0), "MMD": (0.0, 0.5), "EMD": (0.0, 5.0), "FPD": (0.0, 50.0)}
    values = []
    for key, (low, high) in ranges.items():
        value = min(max(float(result[key]), low), high)
        values.append(1.0 - (value - low) / (high - low))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=2000)
    parser.add_argument("--max-real-frames", type=int, default=100)
    args = parser.parse_args()
    manifest = json.loads((args.candidates / "manifest.json").read_text(encoding="utf-8"))
    real_clouds = load_real(manifest["validation_paths"], args.max_real_frames)
    metrics.N_SUBSAMPLE = args.subsample
    results = {}
    for name in manifest["candidates"]:
        candidate = load_candidate(args.candidates / name)
        result = metrics.compute_realism(candidate, real_clouds, PC_RANGE, seed=1234)
        result["normalized_realism"] = normalized_score(result)
        results[name] = result
        print(name, json.dumps(result), flush=True)
    payload = {
        "subsample": args.subsample,
        "real_frames": len(real_clouds),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
