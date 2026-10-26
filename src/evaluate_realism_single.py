#!/usr/bin/env python3
"""Run the official realism metrics for one candidate directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"),
]

import metrics  # noqa: E402
from evaluate_realism_baselines import load_candidate, load_real, normalized_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=10000)
    parser.add_argument("--max-real-frames", type=int, default=100)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    real = load_real(manifest["validation_paths"], args.max_real_frames)
    synthetic = load_candidate(args.candidate)
    metrics.N_SUBSAMPLE = args.subsample
    result = metrics.compute_realism(synthetic, real, (-40, -40, -8, 40, 40, 2), seed=1234)
    result["normalized_realism"] = normalized_score(result)
    result["candidate"] = str(args.candidate)
    result["real_frames"] = len(real)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
