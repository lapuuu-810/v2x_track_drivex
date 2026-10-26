#!/usr/bin/env python3
"""Export a distributional coreset of released UrbanTwin synthetic LiDAR.

Unlike a real-domain mapper, this uses *only* the released synthetic 10K-frame
UrbanTwin source.  Each source frame is represented by a deterministic,
coarsened spherical occupancy histogram.  A greedy kernel-herding selection
then chooses 50 frames whose aggregate histogram approximates the full source
distribution.  Finally, each selected frame is independently subsampled to
the contest size cap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def descriptor(points: np.ndarray, samples: int, seed: int, shape: tuple[int, int, int]) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1)
    valid &= (points >= LOW).all(axis=1) & (points <= HIGH).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        raise ValueError("source frame has no in-range points")
    rng = np.random.default_rng(seed)
    points = points[rng.choice(len(points), min(len(points), samples), replace=False)]
    radius = np.linalg.norm(points, axis=1)
    elevation = np.arctan2(points[:, 2], np.maximum(np.linalg.norm(points[:, :2], axis=1), 1e-6))
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    nr, ne, na = shape
    ir = np.clip((radius / 60.0 * nr).astype(np.int32), 0, nr - 1)
    ie = np.clip(((elevation + 0.35) / 0.70 * ne).astype(np.int32), 0, ne - 1)
    ia = np.clip(((azimuth + np.pi) / (2 * np.pi) * na).astype(np.int32), 0, na - 1)
    code = np.ravel_multi_index((ir, ie, ia), shape)
    hist = np.bincount(code, minlength=nr * ne * na).astype(np.float32)
    return hist / hist.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=9000)
    parser.add_argument("--candidate-stride", type=int, default=10)
    parser.add_argument("--descriptor-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.frames != 50:
        raise ValueError("competition contract needs exactly 50 scans")
    if args.points_per_frame * 12 > 1_500_000:
        raise ValueError("per-frame file would exceed the contest size cap")
    paths = sorted(args.source.glob("*.npy"))
    if len(paths) < args.frames:
        raise ValueError("not enough source frames")
    shape = (24, 12, 32)
    target_sum = np.zeros(np.prod(shape), dtype=np.float64)
    candidate_indices = list(range(0, len(paths), args.candidate_stride))
    candidates = np.empty((len(candidate_indices), np.prod(shape)), dtype=np.float32)
    for index, path in enumerate(paths):
        pts = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1, 3)
        desc = descriptor(pts, args.descriptor_samples, args.seed + index, shape)
        target_sum += desc
        if index % args.candidate_stride == 0:
            candidates[index // args.candidate_stride] = desc
        if (index + 1) % 1000 == 0 or index + 1 == len(paths):
            print(f"described {index + 1}/{len(paths)}", flush=True)
    target = target_sum / len(paths)
    selected: list[int] = []
    selected_sum = np.zeros_like(target)
    available = np.ones(len(candidate_indices), dtype=bool)
    norms = (candidates.astype(np.float64) ** 2).sum(axis=1)
    for step in range(args.frames):
        desired = (step + 1) * target - selected_sum
        loss = norms - 2.0 * (candidates @ desired)
        loss[~available] = np.inf
        chosen = int(np.argmin(loss))
        available[chosen] = False
        selected.append(candidate_indices[chosen])
        selected_sum += candidates[chosen]
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for output_index, source_index in enumerate(selected, 1):
        path = paths[source_index]
        pts = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (pts >= LOW).all(axis=1) & (pts <= HIGH).all(axis=1)
        pts = pts[valid]
        chosen = pts[rng.choice(len(pts), args.points_per_frame, replace=False)]
        chosen.astype(np.float32).tofile(args.output / f"{output_index:04d}.bin")
        rows.append({"source_index": source_index, "source": path.name, "in_range_points": int(len(pts))})
    coreset_mean = selected_sum / args.frames
    report = {
        "method": "synthetic-only spherical histogram kernel-herding coreset",
        "source": str(args.source), "source_frames": len(paths),
        "candidate_stride": args.candidate_stride, "candidate_frames": len(candidate_indices),
        "descriptor_shape": shape, "descriptor_samples": args.descriptor_samples,
        "frames": args.frames, "points_per_frame": args.points_per_frame, "seed": args.seed,
        "synthetic_only": True,
        "l2_error_to_full_native_distribution": float(np.linalg.norm(coreset_mean - target)),
        "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
