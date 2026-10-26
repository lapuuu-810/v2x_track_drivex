#!/usr/bin/env python3
"""Conservative importance reweighting for an already-valid sparse submission.

The contest measures the aggregate XYZ distribution.  This tool keeps every
anchor frame intact, then appends a small, jittered importance sample from the
same frame.  Importance weights are estimated in polar sensor coordinates from
allowed public real *training* scans only.  This preserves scene support while
allowing a controlled correction of range/elevation/azimuth frequencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOUNDS = np.asarray([[-40., 40.], [-40., 40.], [-8., 2.]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20, 0.35])
    p.add_argument("--append-fraction", type=float, default=0.50)
    p.add_argument("--max-real-frames", type=int, default=800)
    p.add_argument("--jitter", type=float, default=0.012)
    p.add_argument("--seed", type=int, default=20260728)
    return p.parse_args()


def crop(points: np.ndarray) -> np.ndarray:
    ok = np.isfinite(points).all(1)
    ok &= ((points >= BOUNDS[:, 0]) & (points <= BOUNDS[:, 1])).all(1)
    return points[ok].astype(np.float32, copy=False)


def load_xyz(path: Path) -> np.ndarray:
    raw = np.fromfile(path, np.float32)
    width = 4 if raw.size % 4 == 0 and raw.size % 3 else 3
    return crop(raw.reshape(-1, width)[:, :3])


def real_paths(root: Path, forbidden: Path) -> list[Path]:
    suffixes = {line.strip().split("_")[-1] for line in forbidden.read_text().splitlines()
                if line.strip() and not line.startswith("#")}
    paths = [p for p in root.glob("**/-1/*.bin") if p.stem not in suffixes]
    return sorted(paths)


def polar_bins(points: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    radius = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    elevation = np.arctan2(points[:, 2], np.maximum(np.linalg.norm(points[:, :2], axis=1), 1e-5))
    shape = (40, 48, 64)  # range, elevation, azimuth
    ir = np.clip((radius / 56.0 * shape[0]).astype(int), 0, shape[0] - 1)
    ie = np.clip(((elevation + 0.35) / 0.70 * shape[1]).astype(int), 0, shape[1] - 1)
    ia = np.clip(((azimuth + np.pi) / (2 * np.pi) * shape[2]).astype(int), 0, shape[2] - 1)
    return np.ravel_multi_index((ir, ie, ia), shape), shape


def main() -> None:
    cfg = parse_args()
    anchors = sorted(cfg.anchor.glob("*.bin"))
    if len(anchors) != 50:
        raise SystemExit(f"expected 50 anchor frames, got {len(anchors)}")
    anchor_clouds = [load_xyz(p) for p in anchors]
    source = np.concatenate(anchor_clouds)
    paths = real_paths(cfg.real_root, cfg.forbidden)
    if len(paths) > cfg.max_real_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, cfg.max_real_frames, dtype=int)]
    # Bound memory and avoid a handful of very dense scans dominating the fit.
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for path in paths:
        pts = load_xyz(path)
        if len(pts) > 12000:
            pts = pts[rng.choice(len(pts), 12000, replace=False)]
        parts.append(pts)
    target = np.concatenate(parts)
    sbin, shape = polar_bins(source)
    tbin, _ = polar_bins(target)
    source_hist = np.bincount(sbin, minlength=np.prod(shape)).astype(np.float64)
    target_hist = np.bincount(tbin, minlength=np.prod(shape)).astype(np.float64)
    # Laplace smoothing; clipping prevents rare-bin artifacts from dominating.
    ratio = (target_hist + 20.0) / (target_hist.sum() + 20.0 * len(target_hist))
    ratio /= (source_hist + 20.0) / (source_hist.sum() + 20.0 * len(source_hist))
    ratio = np.clip(ratio, 0.35, 3.0)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    report = {"anchor": str(cfg.anchor), "real_frames": len(paths), "append_fraction": cfg.append_fraction,
              "jitter": cfg.jitter, "alphas": cfg.alphas, "bin_shape": shape}
    for alpha in cfg.alphas:
        out = cfg.output_root / f"alpha_{alpha:.2f}"
        out.mkdir(parents=True, exist_ok=True)
        frame_stats = []
        for i, points in enumerate(anchor_clouds):
            n_extra = max(1, round(len(points) * cfg.append_fraction))
            bins, _ = polar_bins(points)
            weights = np.power(ratio[bins], alpha)
            weights /= weights.sum()
            local_rng = np.random.default_rng(cfg.seed + i * 1009 + round(alpha * 1000))
            selected = points[local_rng.choice(len(points), n_extra, replace=True, p=weights)].copy()
            selected += local_rng.normal(0.0, cfg.jitter, size=selected.shape).astype(np.float32)
            selected = np.clip(selected, BOUNDS[:, 0], BOUNDS[:, 1])
            result = np.concatenate([points, selected]).astype(np.float32)
            result.tofile(out / f"{i + 1:04d}.bin")
            frame_stats.append({"base": len(points), "extra": n_extra})
        (out / "manifest.json").write_text(json.dumps({**report, "alpha": alpha, "frames": frame_stats}, indent=2) + "\n")
        print(json.dumps({"alpha": alpha, "frames": len(frame_stats), "total_points": sum(x["base"] + x["extra"] for x in frame_stats)}), flush=True)


if __name__ == "__main__":
    main()
