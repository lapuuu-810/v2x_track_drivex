#!/usr/bin/env python3
"""Build a manifest for every non-forbidden linked public validation frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenario_index = {
        path.name: index
        for index, path in enumerate(sorted(p for p in args.full_root.iterdir() if p.is_dir()))
    }
    manifest = []
    for scenario in sorted(p for p in args.allowed_root.iterdir() if p.is_dir()):
        if scenario.name not in scenario_index:
            raise ValueError(f"Unknown scenario {scenario.name}")
        for point_path in sorted((scenario / "-1").glob("*.bin")):
            manifest.append({
                "id": f"{scenario_index[scenario.name]}_{point_path.stem}",
                "path": str(point_path),
                "width": 4,
            })
    ids = [row["id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "frames": len(manifest),
        "scenarios": len({row["id"].split("_")[0] for row in manifest}),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
