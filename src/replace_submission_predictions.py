#!/usr/bin/env python3
"""Build a candidate zip by preserving a validated base submission's non-prediction files."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from build_submission import validate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    frame_ids = [str(value) for value in frame_ids]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    validate_predictions(predictions, frame_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.base, "r") as source, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        names = source.namelist()
        if "predictions.json" not in names:
            raise SystemExit("Base submission has no predictions.json")
        if not any(name.startswith("synthetic/") and name.endswith(".bin") for name in names):
            raise SystemExit("Base submission has no synthetic frames")
        if not any(name in {"declaration.txt", "declaration.pdf"} for name in names):
            raise SystemExit("Base submission has no declaration")
        for info in source.infolist():
            if info.filename != "predictions.json":
                target.writestr(info, source.read(info.filename))
        target.writestr("predictions.json", json.dumps(predictions).encode("utf-8"))
    print(f"Built {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
