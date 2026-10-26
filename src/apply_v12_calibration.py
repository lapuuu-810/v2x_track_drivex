#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROFILES = {
    "vehicle": ((0.98, 1.05, 1.12), 0.05),
    "pedestrian": ((1.0, 1.0, 1.0), 0.0),
    "truck": ((1.35, 1.20, 1.0), 0.25),
}


def calibrate(predictions: dict, pedestrian_filter: bool = True) -> dict:
    output = {}
    for frame_id, rows in predictions.items():
        selected = []
        for source in rows:
            row = copy.deepcopy(source)
            name = row["class"].lower()
            if pedestrian_filter and name == "pedestrian":
                if row["box"][3] > 0.7 or row["box"][4] > 1.2:
                    continue
            scale, z_offset = PROFILES[name]
            row["box"][2] += z_offset
            for index, value in enumerate(scale, start=3):
                row["box"][index] *= value
            selected.append(row)
        output[frame_id] = selected
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if "predictions" in payload:
        payload["predictions"] = calibrate(payload["predictions"])
        output = payload
    else:
        output = calibrate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    predictions = output.get("predictions", output)
    print(json.dumps({
        "frames": len(predictions),
        "detections": sum(len(rows) for rows in predictions.values()),
        "profiles": PROFILES,
        "pedestrian_filter": {"max_length": 0.7, "max_width": 1.2},
    }, indent=2))


if __name__ == "__main__":
    main()
