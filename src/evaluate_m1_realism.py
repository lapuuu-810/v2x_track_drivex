#!/usr/bin/env python3
"""Evaluate an M1 diffusion candidate on its immutable cross-scenario holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "competition" / "starting_kit" / "scoring_program")]
import metrics  # noqa: E402
from evaluate_realism_baselines import load_candidate, load_real, normalized_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=10000)
    args = parser.parse_args()
    corpus = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    validation = [row["source"] for row in corpus["records"] if row["subset"] == "validation"]
    synthetic = load_candidate(args.candidate / "synthetic")
    real = load_real(validation, max_frames=len(validation))
    metrics.N_SUBSAMPLE = args.subsample
    result = metrics.compute_realism(synthetic, real, (-40, -40, -8, 40, 40, 2), seed=1234)
    result["normalized_realism"] = normalized_score(result)
    result.update({"candidate": str(args.candidate), "real_frames": len(real), "protocol": "M1 scenario holdout v1"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
