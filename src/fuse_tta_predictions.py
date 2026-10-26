#!/usr/bin/env python3
"""Fuse TTA predictions with class-wise rotated BEV NMS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GBLOBS = ROOT / "third_party/GBlobs"
sys.path.insert(0, str(GBLOBS))
from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.1)
    parser.add_argument("--max-per-class", type=int, default=100)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    frame_ids = list(data[args.variants[0]])
    output = {}
    for frame_id in frame_ids:
        fused = []
        for name in ("vehicle", "pedestrian", "truck"):
            rows = [
                row for variant in args.variants for row in data[variant][frame_id]
                if row["class"] == name and row["score"] >= args.score_threshold
            ]
            if not rows:
                continue
            boxes = torch.as_tensor(np.asarray([r["box"] for r in rows]), dtype=torch.float32, device="cuda")
            scores = torch.as_tensor([r["score"] for r in rows], dtype=torch.float32, device="cuda")
            keep, _ = iou3d_nms_utils.nms_gpu(
                boxes, scores, args.nms_threshold, pre_maxsize=4096
            )
            for idx in keep[:args.max_per_class].cpu().tolist():
                fused.append(rows[idx])
        output[frame_id] = sorted(fused, key=lambda row: row["score"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output), encoding="utf-8")
    print(json.dumps({"frames": len(output), "detections": sum(map(len, output.values()))}))


if __name__ == "__main__":
    main()
