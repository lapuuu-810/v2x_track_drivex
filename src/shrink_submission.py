#!/usr/bin/env python3
"""Deterministically downsample synthetic frames in a submission zip."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise SystemExit("fraction must be in (0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_before = 0
    total_after = 0
    with zipfile.ZipFile(args.input, "r") as source, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for file_index, info in enumerate(source.infolist()):
            raw = source.read(info.filename)
            if info.filename.startswith("synthetic/") and info.filename.endswith(".bin"):
                points = np.frombuffer(raw, dtype=np.float32)
                if points.size % 3:
                    raise SystemExit(f"Expected XYZ float32 data: {info.filename}")
                points = points.reshape(-1, 3)
                keep_count = max(1, int(round(len(points) * args.fraction)))
                rng = np.random.default_rng(args.seed + file_index)
                indexes = np.sort(rng.choice(len(points), keep_count, replace=False))
                reduced = np.ascontiguousarray(points[indexes], dtype=np.float32)
                total_before += len(points)
                total_after += len(reduced)
                target.writestr(info.filename, reduced.tobytes())
            else:
                target.writestr(info, raw)
    print(
        f"Synthetic points: {total_before} -> {total_after}; "
        f"zip={args.output.stat().st_size / 1024**2:.2f} MiB"
    )


if __name__ == "__main__":
    main()
