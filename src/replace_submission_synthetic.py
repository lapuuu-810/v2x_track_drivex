#!/usr/bin/env python3
"""Replace only synthetic frames in an otherwise validated submission bundle."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = sorted(args.synthetic.glob("*.bin"))
    if len(frames) != 50:
        raise SystemExit(f"Expected exactly 50 synthetic .bin files, found {len(frames)}")
    for path in frames:
        raw = np.fromfile(path, np.float32)
        if raw.size == 0 or raw.size % 3 != 0:
            raise SystemExit(f"{path}: not a non-empty float32 XYZ file")
        if path.stat().st_size > 1_500_000:
            raise SystemExit(f"{path}: exceeds 1.5 MB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.base, "r") as source, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        names = source.namelist()
        if "predictions.json" not in names:
            raise SystemExit("Base submission has no predictions.json")
        if not any(name in {"declaration.txt", "declaration.pdf"} for name in names):
            raise SystemExit("Base submission has no declaration")
        for info in source.infolist():
            if not info.filename.startswith("synthetic/"):
                target.writestr(info, source.read(info.filename))
        for index, path in enumerate(frames, 1):
            target.write(path, f"synthetic/{index:04d}.bin")
    print(f"Built {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
