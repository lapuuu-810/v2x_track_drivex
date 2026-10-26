#!/usr/bin/env python3
"""Compose per-class detector outputs and apply validation-selected calibration.

Example:
  python src/compose_predictions.py \
    --frame-ids competition/starting_kit/detection_test_frame_ids.json \
    --source vehicle=outputs/car.json \
    --source pedestrian=outputs/gblobs.json \
    --source truck=outputs/gblobs.json \
    --threshold vehicle=0.05 --threshold pedestrian=0.03 --threshold truck=0.10 \
    --scale vehicle=0.98,1.05,1.12 --z-offset vehicle=0.05 \
    --output outputs/predictions_composed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASSES = ("vehicle", "pedestrian", "truck")


def parse_key_values(items, value_parser, default):
    result = dict(default)
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or key not in CLASSES:
            raise ValueError(f"Expected CLASS=VALUE, with CLASS in {CLASSES}: {item}")
        result[key] = value_parser(value)
    return result


def parse_scale(value):
    values = tuple(float(part) for part in value.split(","))
    if len(values) != 3 or any(part <= 0 for part in values):
        raise ValueError(f"Scale must be three positive values dx,dy,dz: {value}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="CLASS=JSON")
    parser.add_argument("--threshold", action="append", default=[], metavar="CLASS=FLOAT")
    parser.add_argument("--scale", action="append", default=[], metavar="CLASS=DX,DY,DZ")
    parser.add_argument("--z-offset", action="append", default=[], metavar="CLASS=FLOAT")
    parser.add_argument("--score-scale", action="append", default=[], metavar="CLASS=FLOAT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = parse_key_values(args.source, Path, {})
    missing_sources = [name for name in CLASSES if name not in sources]
    if missing_sources:
        raise SystemExit(f"Missing per-class sources: {missing_sources}")
    thresholds = parse_key_values(args.threshold, float, {name: 0.0 for name in CLASSES})
    scales = parse_key_values(args.scale, parse_scale, {name: (1.0, 1.0, 1.0) for name in CLASSES})
    z_offsets = parse_key_values(args.z_offset, float, {name: 0.0 for name in CLASSES})
    score_scales = parse_key_values(args.score_scale, float, {name: 1.0 for name in CLASSES})

    ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    frame_ids = [str(value) for value in frame_ids]
    loaded = {
        class_name: json.loads(path.read_text(encoding="utf-8"))
        for class_name, path in sources.items()
    }
    expected = set(frame_ids)
    for class_name, predictions in loaded.items():
        if set(predictions) != expected:
            raise SystemExit(f"{class_name} source frame IDs do not match canonical IDs")

    output = {frame_id: [] for frame_id in frame_ids}
    counts = {name: 0 for name in CLASSES}
    for frame_id in frame_ids:
        for class_name in CLASSES:
            dx_scale, dy_scale, dz_scale = scales[class_name]
            for row in loaded[class_name][frame_id]:
                if str(row.get("class", "vehicle")).lower() != class_name:
                    continue
                score = float(row["score"])
                if score < thresholds[class_name]:
                    continue
                box = [float(value) for value in row["box"]]
                if len(box) != 7:
                    raise SystemExit(f"{frame_id}: box must have seven values")
                box[2] += z_offsets[class_name]
                box[3] *= dx_scale
                box[4] *= dy_scale
                box[5] *= dz_scale
                output[frame_id].append({
                    "box": box,
                    "score": min(1.0, max(0.0, score * score_scales[class_name])),
                    "class": class_name,
                })
                counts[class_name] += 1
        output[frame_id].sort(key=lambda row: row["score"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(output), "detections": counts}, indent=2))


if __name__ == "__main__":
    main()
