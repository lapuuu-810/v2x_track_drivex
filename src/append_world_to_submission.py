#!/usr/bin/env python3
"""Keep base submission predictions byte-for-byte and append only novel rows."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--expanded-world", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--base-score-floor", type=float, default=0.05)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_zip) as source:
        base_bytes = source.read("predictions.json")
        base = json.loads(base_bytes)
    expanded = json.loads(args.expanded_world.read_text())
    output = {}
    added = 0
    for frame_id, base_rows in base.items():
        novel = [
            row for row in expanded[frame_id]
            if float(row["score"]) < args.base_score_floor
        ]
        output[frame_id] = base_rows + novel
        output[frame_id].sort(key=lambda row: float(row["score"]), reverse=True)
        added += len(novel)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, separators=(",", ":")))
    with zipfile.ZipFile(args.base_zip) as source, zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        for info in source.infolist():
            if info.filename != "predictions.json":
                target.writestr(info, source.read(info.filename))
        target.write(args.output_json, "predictions.json")
    print(json.dumps({
        "frames": len(output),
        "base": sum(map(len, base.values())),
        "added": added,
        "total": sum(map(len, output.values())),
    }))


if __name__ == "__main__":
    main()
