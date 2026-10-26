#!/usr/bin/env python3
"""Fit a LiDM range calibration LUT using *only* the M1 training split.

This learns a per-ring monotone mapping from autoencoder output log-range to
real log-range.  It is a model calibration: validation or hidden frames are
never read during fitting.  The saved LUT can consequently be applied blindly
to reconstructed or sampled scans at export time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from export_lidm_reconstruction import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quantiles", type=int, default=257)
    parser.add_argument("--method", choices=("quantile", "isotonic"), default="quantile")
    parser.add_argument("--max-per-ring", type=int, default=120000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.quantiles < 3:
        raise SystemExit("--quantiles must be at least 3")

    paths = sorted((args.corpus / "train").glob("*.npz"))
    if not paths:
        raise SystemExit(f"No train scans in {args.corpus}")
    model = load_model(args.config, args.checkpoint, torch.device(args.device))
    rng = np.random.default_rng(20260728)
    pred_parts, true_parts = [[] for _ in range(64)], [[] for _ in range(64)]
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            records = [np.load(path) for path in batch_paths]
            try:
                ranges = np.stack([x["log_range"].astype(np.float32) for x in records])
                masks = np.stack([x["mask"].astype(bool) for x in records])
            finally:
                for x in records:
                    x.close()
            x_range = ranges * 2.0 - 1.0
            if model.encoder.conv_in.in_channels == 2:
                model_input = np.stack((x_range, masks.astype(np.float32) * 2.0 - 1.0), axis=1)
            else:
                model_input = x_range[:, None]
            reconstruction, _ = model(torch.from_numpy(model_input).to(args.device))
            predicted = np.clip((reconstruction[:, 0].float().cpu().numpy() + 1.0) * .5, 0., 1.)
            for ring in range(64):
                keep = masks[:, ring, :]
                p, t = predicted[:, ring, :][keep], ranges[:, ring, :][keep]
                if len(p):
                    pred_parts[ring].append(p)
                    true_parts[ring].append(t)
            print(f"encoded {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)

    qs = np.linspace(0., 1., args.quantiles, dtype=np.float64)
    knots = np.empty((64, args.quantiles), dtype=np.float32)
    values = np.empty_like(knots)
    counts = []
    for ring in range(64):
        p, t = np.concatenate(pred_parts[ring]), np.concatenate(true_parts[ring])
        if len(p) > args.max_per_ring:
            pick = rng.choice(len(p), size=args.max_per_ring, replace=False)
            p, t = p[pick], t[pick]
        if args.method == "quantile":
            # Marginal quantile matching corrects the range distribution.
            k = np.quantile(p, qs)
            v = np.quantile(t, qs)
        else:
            # Conditional binned means use paired train reconstructions; PAVA
            # makes the mapping monotone without involving validation frames.
            k = qs.copy()
            idx = np.minimum((p * (args.quantiles - 1)).astype(np.int64), args.quantiles - 1)
            weight = np.bincount(idx, minlength=args.quantiles).astype(np.float64)
            sums = np.bincount(idx, weights=t, minlength=args.quantiles)
            observed = weight > 0
            mean = np.empty(args.quantiles, dtype=np.float64)
            mean[observed] = sums[observed] / weight[observed]
            mean[~observed] = np.interp(k[~observed], k[observed], mean[observed])
            # Pool-adjacent-violators algorithm (weighted isotonic regression).
            blocks = []
            for j, (value, w) in enumerate(zip(mean, np.maximum(weight, 1e-3))):
                blocks.append([j, j, float(value), float(w)])
                while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
                    left, right = blocks[-2], blocks[-1]
                    tw = left[3] + right[3]
                    blocks[-2:] = [[left[0], right[1], (left[2] * left[3] + right[2] * right[3]) / tw, tw]]
            v = np.empty(args.quantiles, dtype=np.float64)
            for first, last, value, _ in blocks:
                v[first:last + 1] = value
        knots[ring] = k.astype(np.float32)
        values[ring] = v.astype(np.float32)
        knots[ring, 0], knots[ring, -1] = 0., 1.
        values[ring] = np.clip(np.maximum.accumulate(values[ring]), 0., 1.)
        values[ring, 0], values[ring, -1] = 0., 1.
        if args.method == "quantile":
            for j in range(1, len(knots[ring])):
                knots[ring, j] = max(knots[ring, j], knots[ring, j - 1] + 1e-6)
            knots[ring, -1] = 1.0
        counts.append(int(len(p)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, knots=knots, values=values, quantiles=qs.astype(np.float32), counts=np.asarray(counts), method=args.method)
    print({"output": str(args.output), "method": args.method, "rings": 64, "samples_per_ring_min": min(counts), "samples_per_ring_max": max(counts)})


if __name__ == "__main__":
    main()
