#!/usr/bin/env python3
"""Run an exact GBlobs YAML/checkpoint on canonical challenge frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GBLOBS_ROOT = PROJECT_ROOT / "third_party" / "GBlobs"
sys.path[:0] = [str(GBLOBS_ROOT / "tools"), str(GBLOBS_ROOT)]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from common import read_points  # noqa: E402


class ChallengeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, paths, frame_ids, binary_width, logger):
        super().__init__(dataset_cfg, class_names, False, Path("."), logger)
        self.paths = paths
        self.frame_ids = frame_ids
        self.binary_width = binary_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        points = read_points(self.paths[index], self.binary_width)
        points = points[np.any(points[:, :3] != 0, axis=1)]
        return self.prepare_data({"points": points, "frame_id": self.frame_ids[index]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-width", type=int, choices=(3, 4), default=3)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    args = parser.parse_args()

    cfg_from_yaml_file(str(args.config), cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_threshold
    ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    frame_ids = [str(value) for value in frame_ids]
    files = sorted(path for path in args.frames.iterdir() if path.suffix.lower() in {".bin", ".npy"})
    by_stem = {path.stem: path for path in files}
    if not all(frame_id in by_stem for frame_id in frame_ids):
        raise SystemExit("Challenge point-cloud files do not match canonical frame IDs")

    logger = common_utils.create_logger()
    dataset = ChallengeDataset(
        cfg.DATA_CONFIG,
        cfg.CLASS_NAMES,
        [by_stem[frame_id] for frame_id in frame_ids],
        frame_ids,
        args.binary_width,
        logger,
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(args.checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()

    predictions = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            pred = model.forward(batch)[0][0]
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            predictions[frame_ids[index]] = [
                {
                    "box": [float(value) for value in box[:7]],
                    "score": float(np.clip(score, 0, 1)),
                    "class": cfg.CLASS_NAMES[int(label) - 1],
                }
                for box, score, label in zip(boxes, scores, labels)
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    counts = {name: 0 for name in cfg.CLASS_NAMES}
    for rows in predictions.values():
        for row in rows:
            counts[row["class"]] += 1
    print(json.dumps({"frames": len(predictions), "detections": counts}, indent=2))


if __name__ == "__main__":
    main()
