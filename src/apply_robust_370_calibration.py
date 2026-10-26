#!/usr/bin/env python3
"""Apply box calibration selected on all 370 non-forbidden validation frames."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROFILES = {
    "vehicle": ((1.00, 1.10, 1.15), 0.05),
    "pedestrian": ((1.00, 1.00, 1.00), 0.00),
    "truck": ((1.40, 1.10, 1.10), 0.20),
}


def calibrate(predictions: dict) -> dict:
    output = {}
    for frame_id, rows in predictions.items():
        selected = []
        for source in rows:
            row = copy.deepcopy(source)
            name = row["class"].lower()
            if name == "pedestrian" and (row["box"][3] > 1.2 or row["box"][4] > 1.2):
                continue
            scale, z_offset = PROFILES[name]
            row["box"][2] += z_offset
            row["box"][3] *= scale[0]
            row["box"][4] *= scale[1]
            row["box"][5] *= scale[2]
            selected.append(row)
        output[frame_id] = selected
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    output = calibrate(payload.get("predictions", payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "frames": len(output),
        "detections": sum(map(len, output.values())),
        "profiles": PROFILES,
    }))


if __name__ == "__main__":
    main()
