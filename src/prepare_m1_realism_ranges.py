#!/usr/bin/env python3
"""Prepare a cross-scenario M1 LiDAR range-image corpus for realism generation.

The challenge's hidden realism reference has roughly 50k returns per scan.  The
public M1 sensor is a native 64 x 1024 organised LiDAR, unlike the 128 x 1024
M2 source used by the earlier RangeLDM.  This script deliberately keeps only
log-range and return masks, makes an immutable scenario holdout, and records
every source path for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

H, W = 64, 1024
PC_MIN = np.asarray([-40.0, -40.0, -8.0], np.float32)
PC_MAX = np.asarray([40.0, 40.0, 2.0], np.float32)


def bucket(value: str, modulo: int) -> int:
    return int.from_bytes(hashlib.sha1(value.encode()).digest()[:8], "big") % modulo


def choose_layout(raw: np.ndarray) -> np.ndarray:
    candidates = [raw.reshape(H, W, 4), raw.reshape(W, H, 4).transpose(1, 0, 2)]
    scores = []
    for grid in candidates:
        xyz = grid[..., :3]
        valid = np.isfinite(xyz).all(-1) & np.any(xyz != 0, axis=-1)
        elevation = np.arctan2(xyz[..., 2], np.linalg.norm(xyz[..., :2], axis=-1))
        row_var = [np.var(elevation[row, valid[row]]) if valid[row].any() else 1.0 for row in range(H)]
        scores.append(float(np.median(row_var)))
    return candidates[int(np.argmin(scores))]


def encode(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    raw = np.fromfile(path, np.float32)
    if raw.size != H * W * 4:
        raise ValueError(f"{path}: expected {H}x{W} xyzi, got {raw.size // 4} points")
    grid = choose_layout(raw)
    xyz = grid[..., :3]
    valid = np.isfinite(xyz).all(-1) & np.any(xyz != 0, axis=-1)
    valid &= (xyz >= PC_MIN).all(-1) & (xyz <= PC_MAX).all(-1)
    ranges = np.linalg.norm(xyz, axis=-1)
    log_range = np.zeros((H, W), np.float16)
    log_range[valid] = (np.log2(ranges[valid] + 1.0) / 8.0).astype(np.float16)
    elevation = np.arctan2(xyz[..., 2], np.linalg.norm(xyz[..., :2], axis=-1))
    row_angles = np.asarray([
        np.median(elevation[row, valid[row]]) if valid[row].any() else np.nan for row in range(H)
    ], np.float32)
    return {"log_range": log_range, "mask": valid.astype(np.uint8)}, row_angles, int(valid.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.holdout_bucket < args.holdout_modulo:
        raise SystemExit("holdout-bucket must be in [0, holdout-modulo)")
    forbidden = {
        line.strip().split("_")[-1]
        for line in args.forbidden.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    paths = [p for p in sorted(args.real_root.glob("**/-1/*.bin")) if p.stem not in forbidden]
    if not paths:
        raise SystemExit("No eligible M1 frames")
    args.output.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "validation"):
        (args.output / subset).mkdir(exist_ok=True)
    records, angle_rows = [], []
    for index, path in enumerate(paths):
        scenario = path.parent.parent.name
        subset = "validation" if bucket(scenario, args.holdout_modulo) == args.holdout_bucket else "train"
        payload, angles, returns = encode(path)
        output_name = f"{index:06d}_{scenario}_{path.stem}.npz"
        np.savez_compressed(args.output / subset / output_name, **payload)
        angle_rows.append(angles)
        records.append({"source": str(path), "scenario": scenario, "subset": subset, "output": output_name, "returns": returns})
        if (index + 1) % 50 == 0 or index + 1 == len(paths):
            print(f"prepared {index + 1}/{len(paths)}", flush=True)
    angles = np.nanmedian(np.stack(angle_rows), axis=0).astype(np.float32)
    if not np.isfinite(angles).all():
        raise RuntimeError("Could not estimate all M1 laser elevations")
    np.save(args.output / "ring_angles.npy", angles)
    summary = {
        "method": "M1 native 64x1024 log-range/mask corpus with scenario holdout",
        "real_root": str(args.real_root),
        "forbidden_policy": "exclude every numeric suffix occurring in forbidden_frames.txt",
        "holdout": {"modulo": args.holdout_modulo, "bucket": args.holdout_bucket},
        "shape": [H, W],
        "train_frames": sum(row["subset"] == "train" for row in records),
        "validation_frames": sum(row["subset"] == "validation" for row in records),
        "median_returns": int(np.median([row["returns"] for row in records])),
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
