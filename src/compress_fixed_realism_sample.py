#!/usr/bin/env python3
"""Compress realism frames while preserving the official fixed-seed sample.

The scorer samples synthetic points and real points with the same RNG.  Merely
keeping the selected 10k synthetic points changes the RNG state before real
sampling.  This script instead keeps the original in-range point count and
index space, fills unselected positions with a highly compressible in-range
point, and restores the selected values at exactly the indices chosen by the
official seed.  Thus both sampled point sets and all subsequent RNG state are
unchanged.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np


PC_RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)


def load_xyz(raw: bytes, name: str) -> np.ndarray:
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size == 0 or values.size % 3:
        raise ValueError(f"{name}: expected non-empty float32 XYZ")
    return values.reshape(-1, 3)


def in_range(points: np.ndarray) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = PC_RANGE
    return (
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-size", type=int, default=10_000)
    args = parser.parse_args()

    with zipfile.ZipFile(args.input) as source:
        syn_names = sorted(
            n for n in source.namelist()
            if n.startswith("synthetic/") and n.endswith(".bin")
        )
        if len(syn_names) != 50:
            raise SystemExit(f"Expected 50 synthetic files, found {len(syn_names)}")
        clouds = [load_xyz(source.read(n), n) for n in syn_names]
        filtered = [p[in_range(p)] for p in clouds]
        counts = np.asarray([len(p) for p in filtered], dtype=np.int64)
        original = np.concatenate(filtered, axis=0)
        if len(original) < args.sample_size:
            raise SystemExit("This compressor requires at least sample-size in-range points")

        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(original), args.sample_size, replace=False)
        compact = np.zeros_like(original)
        compact[:, 2] = np.float32(-7.999)  # valid, compressible filler
        compact[selected] = original[selected]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fixed-realism-") as temp_dir:
            temp = Path(temp_dir)
            offset = 0
            paths = []
            for index, count in enumerate(counts, 1):
                path = temp / f"{index:04d}.bin"
                compact[offset:offset + count].tofile(path)
                offset += count
                paths.append(path)

            with zipfile.ZipFile(
                args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as target:
                for info in source.infolist():
                    if not info.filename.startswith("synthetic/"):
                        target.writestr(info, source.read(info.filename))
                for path in paths:
                    target.write(path, f"synthetic/{path.name}")

    ratio = args.output.stat().st_size / args.input.stat().st_size
    print(f"in-range points: {len(original):,}")
    print(f"preserved sample: {len(selected):,}")
    print(f"output: {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB, {ratio:.3f}x)")


if __name__ == "__main__":
    main()
