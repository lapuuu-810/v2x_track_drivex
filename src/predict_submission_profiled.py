#!/usr/bin/env python3
"""Run a checkpoint with its exact runtime profile on challenge frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"
sys.path[:0] = [str(OPENPCDET_ROOT / "tools"), str(OPENPCDET_ROOT)]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from common import read_points  # noqa: E402


def apply_profile(profile: str, score_threshold: float):
    cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES = 384
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = score_threshold
    cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS = False
    cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = 0.01
    vehicle = {
        "class_name": "vehicle",
        "anchor_sizes": [[4.67, 2.01, 1.52]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-3.04],
        "align_center": True,
        "feature_map_stride": 8,
        "matched_threshold": 0.55,
        "unmatched_threshold": 0.4,
    }
    if profile == "car":
        cfg.CLASS_NAMES = ["vehicle"]
        cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = [vehicle]
        return
    if profile == "three_class":
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]
        cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = [
            vehicle,
            {
                "class_name": "pedestrian",
                "anchor_sizes": [[1.85, 0.77, 1.62], [0.65, 0.60, 1.70]],
                "anchor_rotations": [0, 1.57],
                "anchor_bottom_heights": [-2.35],
                "align_center": True,
                "feature_map_stride": 8,
                "matched_threshold": 0.45,
                "unmatched_threshold": 0.3,
            },
            {
                "class_name": "truck",
                "anchor_sizes": [[5.2, 2.1, 2.1], [8.0, 2.8, 3.2]],
                "anchor_rotations": [0, 1.57],
                "anchor_bottom_heights": [-3.30],
                "align_center": True,
                "feature_map_stride": 8,
                "matched_threshold": 0.5,
                "unmatched_threshold": 0.35,
            },
        ]
        return
    raise ValueError(profile)


class ChallengeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, paths, ids, binary_width, logger):
        super().__init__(dataset_cfg, class_names, False, Path("."), logger)
        self.paths = paths
        self.ids = ids
        self.binary_width = binary_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        points = read_points(self.paths[index], self.binary_width)
        # Released challenge frames are padded to a fixed length with XYZ=0.
        points = points[np.any(points[:, :3] != 0, axis=1)]
        return self.prepare_data({"points": points, "frame_id": self.ids[index]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("car", "three_class"), required=True)
    parser.add_argument("--binary-width", type=int, choices=(3, 4), default=3)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    args = parser.parse_args()

    cfg_from_yaml_file(str(args.config), cfg)
    apply_profile(args.profile, args.score_threshold)
    ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    frame_ids = [str(x) for x in frame_ids]
    files = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in {".bin", ".npy"})
    by_stem = {p.stem: p for p in files}
    if not all(frame_id in by_stem for frame_id in frame_ids):
        raise SystemExit("Challenge point-cloud files do not match canonical frame IDs")
    paths = [by_stem[frame_id] for frame_id in frame_ids]

    logger = common_utils.create_logger()
    dataset = ChallengeDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, paths, frame_ids, args.binary_width, logger)
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
                    "box": [float(x) for x in box[:7]],
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
