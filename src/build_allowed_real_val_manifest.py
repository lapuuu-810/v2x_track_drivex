#!/usr/bin/env python3
"""Freeze a non-forbidden public V2X-Real train-split validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    forbidden = {
        row.strip().split("_")[-1]
        for row in args.forbidden.read_text().splitlines()
        if row.strip() and not row.startswith("#")
    }
    records = []
    for path in sorted(args.real_root.glob("**/-1/*.bin")):
        if path.stem in forbidden:
            continue
        scene = path.parent.parent.name
        records.append({
            "id": f"{scene}__{path.stem}",
            "scene": scene,
            "frame": path.stem,
            "path": str(path.resolve()),
        })
    ranked = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest(),
    )
    chosen = sorted(ranked[: args.count], key=lambda row: row["id"])
    if len(chosen) != args.count or len({row["id"] for row in chosen}) != args.count:
        raise ValueError(f"expected {args.count} unique allowed records, found {len(chosen)}")
    manifest = {
        "purpose": "fixed public-real validation only; never detector training",
        "real_root": str(args.real_root.resolve()),
        "forbidden_file": str(args.forbidden.resolve()),
        "forbidden_policy": "exclude every frame whose numeric stem matches any forbidden frame suffix",
        "seed": args.seed,
        "available_allowed_frames": len(records),
        "selected_frames": len(chosen),
        "records": chosen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
