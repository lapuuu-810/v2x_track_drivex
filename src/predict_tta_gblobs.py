#!/usr/bin/env python3
"""Run geometric test-time augmentation and map detections back to source space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GBLOBS = ROOT / "third_party/GBlobs"
sys.path[:0] = [str(GBLOBS / "tools"), str(GBLOBS)]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


class OneFrameDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, logger):
        super().__init__(dataset_cfg, class_names, False, Path("."), logger)

    def prepare(self, points: np.ndarray, frame_id: str):
        return self.prepare_data({"points": points, "frame_id": frame_id})


def transform_points(points: np.ndarray, variant: str) -> np.ndarray:
    out = points.copy()
    if variant == "xflip":
        out[:, 1] *= -1
    elif variant == "yflip":
        out[:, 0] *= -1
    elif variant.startswith("rot"):
        angle = np.deg2rad(float(variant[3:]))
        c, s = np.cos(angle), np.sin(angle)
        xy = out[:, :2].copy()
        out[:, 0] = xy[:, 0] * c - xy[:, 1] * s
        out[:, 1] = xy[:, 0] * s + xy[:, 1] * c
    elif variant != "identity":
        raise ValueError(f"Unknown variant: {variant}")
    return out


def inverse_boxes(boxes: np.ndarray, variant: str) -> np.ndarray:
    out = boxes.copy()
    if variant == "xflip":
        out[:, 1] *= -1
        out[:, 6] *= -1
    elif variant == "yflip":
        out[:, 0] *= -1
        out[:, 6] = -(out[:, 6] + np.pi)
    elif variant.startswith("rot"):
        angle = np.deg2rad(float(variant[3:]))
        c, s = np.cos(angle), np.sin(angle)
        xy = out[:, :2].copy()
        out[:, 0] = xy[:, 0] * c + xy[:, 1] * s
        out[:, 1] = -xy[:, 0] * s + xy[:, 1] * c
        out[:, 6] -= angle
    out[:, 6] = (out[:, 6] + np.pi) % (2 * np.pi) - np.pi
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["identity", "xflip", "yflip"])
    parser.add_argument("--score-threshold", type=float, default=0.01)
    args = parser.parse_args()

    cfg_from_yaml_file(str(args.config), cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_threshold
    logger = common_utils.create_logger()
    dataset = OneFrameDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(args.checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()
    manifest = json.loads(args.manifest.read_text())

    result = {variant: {} for variant in args.variants}
    with torch.no_grad():
        for index, item in enumerate(manifest):
            points = np.fromfile(item["path"], dtype=np.float32).reshape(-1, item.get("width", 4))
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[1] == 3:
                points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
            points[:, 3] = 0
            for variant in args.variants:
                batch = dataset.collate_batch([dataset.prepare(transform_points(points, variant), item["id"])])
                load_data_to_gpu(batch)
                pred = model(batch)[0][0]
                boxes = inverse_boxes(pred["pred_boxes"].cpu().numpy(), variant)
                scores = pred["pred_scores"].cpu().numpy()
                labels = pred["pred_labels"].cpu().numpy()
                result[variant][item["id"]] = [
                    {"box": box[:7].tolist(), "score": float(score),
                     "class": cfg.CLASS_NAMES[int(label) - 1]}
                    for box, score, label in zip(boxes, scores, labels)
                ]
            if (index + 1) % 10 == 0:
                logger.info("TTA inference %d/%d", index + 1, len(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
