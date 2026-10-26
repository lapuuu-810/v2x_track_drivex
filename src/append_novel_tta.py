#!/usr/bin/env python3
"""Append novel TTA boxes without replacing or re-ranking base detections."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/GBlobs"))
from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402

from apply_v14_voxelrcnn_calibration import calibrate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--variant", default="rot-5")
    parser.add_argument("--source-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.1)
    parser.add_argument("--tail-min", type=float, default=0.001)
    parser.add_argument("--tail-max", type=float, default=0.049)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    variants = json.loads(args.variants.read_text())
    candidate = calibrate(variants[args.variant])
    output = copy.deepcopy(base)
    added = {name: 0 for name in ("vehicle", "pedestrian", "truck")}

    for frame_id, base_rows in base.items():
        for name in added:
            fixed = [row for row in base_rows if row["class"] == name]
            novel = [
                row for row in candidate[frame_id]
                if row["class"] == name and row["score"] >= args.source_threshold
            ]
            if not novel:
                continue
            rows = fixed + novel
            boxes = torch.as_tensor(
                np.asarray([row["box"] for row in rows]),
                dtype=torch.float32,
                device="cuda",
            )
            scores = torch.as_tensor(
                [2.0 + row["score"] for row in fixed]
                + [row["score"] for row in novel],
                dtype=torch.float32,
                device="cuda",
            )
            keep, _ = iou3d_nms_utils.nms_gpu(
                boxes, scores, args.nms_threshold, pre_maxsize=4096
            )
            novel_indices = [index - len(fixed) for index in keep.cpu().tolist()
                             if index >= len(fixed)]
            selected = [copy.deepcopy(novel[index]) for index in novel_indices]
            selected.sort(key=lambda row: row["score"], reverse=True)
            count = len(selected)
            for rank, row in enumerate(selected):
                fraction = 1.0 if count == 1 else 1.0 - rank / (count - 1)
                row["score"] = args.tail_min + (
                    args.tail_max - args.tail_min
                ) * fraction
                output[frame_id].append(row)
                added[name] += 1
        output[frame_id].sort(key=lambda row: row["score"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output))
    print(json.dumps({
        "frames": len(output),
        "base": sum(map(len, base.values())),
        "added": added,
        "total": sum(map(len, output.values())),
    }))


if __name__ == "__main__":
    main()
