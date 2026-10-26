#!/usr/bin/env python3
"""Convert synthetic XYZ anchors to static-map conditions for the renderer.

Unlike the training-pair preparer, this utility reads *only* submitted
synthetic geometry.  It never opens a real point cloud.  Each output map is
therefore a geometry condition for a learned sensor renderer, not a disguised
reference cloud.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOW = np.asarray([-40.0, -40.0, -8.0], np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], np.float32)


def range_grid(xyz: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(xyz).all(1) & np.any(xyz != 0, 1)
    keep &= (xyz >= LOW).all(1) & (xyz <= HIGH).all(1)
    xyz = xyz[keep]
    out = np.zeros((height, width), np.float32)
    if len(xyz):
        radial = np.linalg.norm(xyz[:, :2], axis=1)
        ranges = np.linalg.norm(xyz, axis=1)
        elev = np.arctan2(xyz[:, 2], np.maximum(radial, 1e-6))
        azim = np.arctan2(xyz[:, 1], xyz[:, 0])
        iy = np.clip(((elev + .35) / .70 * height).astype(np.int32), 0, height - 1)
        ix = np.clip(((azim + np.pi) / (2 * np.pi) * width).astype(np.int32), 0, width - 1)
        flat = iy.astype(np.int64) * width + ix
        order = np.argsort(ranges, kind="stable")
        _, first = np.unique(flat[order], return_index=True)
        chosen = order[first]
        out[iy[chosen], ix[chosen]] = ranges[chosen]
    mask = (out > 0).astype(np.float32)
    log = np.zeros_like(out)
    log[mask > 0] = np.log2(out[mask > 0] + 1.0) / 8.0
    return log, mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=1024)
    a = p.parse_args()
    frames = sorted(a.synthetic.glob("*.bin"))
    if len(frames) != 50:
        raise SystemExit(f"expected 50 XYZ frames, got {len(frames)}")
    target = a.output / "train"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(frames, 1):
        raw = np.fromfile(path, np.float32)
        if raw.size % 3:
            raise SystemExit(f"{path}: expected float32 XYZ")
        log, mask = range_grid(raw.reshape(-1, 3), a.height, a.width)
        dst = target / f"{i:04d}.npz"
        np.savez_compressed(dst, static_log=log, static_mask=mask)
        rows.append({"frame": path.name, "condition": str(dst), "input_points": int(raw.size // 3), "occupied_cells": int(mask.sum())})
    report = {
        "method": "synthetic-only per-frame range-map conditioning",
        "policy": "only --synthetic frames are read; no real/reference frames are accessed",
        "shape": [a.height, a.width],
        "frames": rows,
    }
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames": len(rows), "mean_occupied_cells": float(np.mean([r["occupied_cells"] for r in rows]))}, indent=2))


if __name__ == "__main__":
    main()
