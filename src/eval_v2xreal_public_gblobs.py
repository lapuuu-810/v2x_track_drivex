#!/usr/bin/env python3
"""Evaluate a GBlobs checkpoint on the public V2X-Real IC validation split.

Public labels are validation-only. This program never writes them into the
synthetic training set or database sampler.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GBLOBS_ROOT = PROJECT_ROOT / "third_party" / "GBlobs"
V2XREAL_ROOT = PROJECT_ROOT / "third_party" / "V2X-Real"
SCORING_ROOT = PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"
sys.path[:0] = [
    str(GBLOBS_ROOT / "tools"),
    str(GBLOBS_ROOT),
    str(V2XREAL_ROOT),
    str(SCORING_ROOT),
]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils import box_utils  # noqa: E402
from detection_eval import evaluate_3d_detection  # noqa: E402


PC_RANGE = [-40, -40, -8, 40, 40, 2]
INVERSE_CLASS_MAP = {
    fine_name: super_name
    for super_name, fine_names in SUPER_CLASS_MAP.items()
    for fine_name in fine_names
}


class PublicICDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, samples, logger):
        super().__init__(dataset_cfg, class_names, False, Path("."), logger)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        points = np.fromfile(sample["bin"], dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        # Challenge input is XYZ. Suppress public-only return-count intensity.
        points[:, 3] = 0.0
        return self.prepare_data({"points": points, "frame_id": sample["id"]})


def discover_ic_samples(root: Path, stride: int, max_frames: int | None):
    samples = []
    scenarios = sorted(path for path in root.iterdir() if path.is_dir())
    for scenario_index, scenario in enumerate(scenarios):
        agents = sorted(
            (path for path in scenario.iterdir() if path.is_dir() and int(path.name) < 0),
            key=lambda path: path.name,
        )
        if not agents:
            continue
        ego = agents[0]
        yamls = sorted(path for path in ego.glob("*.yaml") if "additional" not in path.name)
        for yaml_path in yamls[::stride]:
            bin_path = yaml_path.with_suffix(".bin")
            if not bin_path.exists():
                continue
            samples.append({
                "id": f"{scenario_index}_{yaml_path.stem}",
                "yaml": yaml_path,
                "bin": bin_path,
            })
            if max_frames is not None and len(samples) >= max_frames:
                return samples
    return samples


def ground_truth_for_sample(sample):
    metadata = yaml.safe_load(sample["yaml"].read_text(encoding="utf-8"))
    grouped = {}
    for object_id, obj in metadata.get("vehicles", {}).items():
        super_name = INVERSE_CLASS_MAP.get(obj.get("obj_type"))
        if super_name is None:
            continue
        obj_copy = copy.deepcopy(obj)
        obj_copy["obj_type"] = np.asarray([super_name])
        grouped[object_id] = obj_copy

    projected = {}
    box_utils.project_world_objects(grouped, projected, metadata["lidar_pose"], PC_RANGE, "lwh")
    return [
        {"box": [float(value) for value in value[0][:7]], "class": str(value[0][7]).lower()}
        for value in projected.values()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    )
    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.config), cfg)
    # Cache low-score candidates once; thresholding below is CPU-only.
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = min(args.thresholds)
    samples = discover_ic_samples(args.data_root, args.stride, args.max_frames)
    if not samples:
        raise SystemExit("No infrastructure validation samples found")
    logger.info("Discovered %d public IC validation samples", len(samples))

    dataset = PublicICDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, samples, logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(args.checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()

    predictions = {}
    ground_truth = {}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            pred = model.forward(batch)[0][0]
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            predictions[sample["id"]] = [
                {
                    "box": [float(value) for value in box[:7]],
                    "score": float(score),
                    "class": cfg.CLASS_NAMES[int(label) - 1],
                }
                for box, score, label in zip(boxes, scores, labels)
            ]
            ground_truth[sample["id"]] = ground_truth_for_sample(sample)
            if (index + 1) % 50 == 0:
                logger.info("Inference %d/%d", index + 1, len(samples))

    class_names = list(cfg.CLASS_NAMES)
    threshold_results = {}
    for threshold in args.thresholds:
        filtered = {
            frame_id: [row for row in rows if row["score"] >= threshold]
            for frame_id, rows in predictions.items()
        }
        threshold_results[str(threshold)] = {
            "3d": evaluate_3d_detection(filtered, ground_truth, class_names, (0.5, 0.7), "3d"),
            "bev": evaluate_3d_detection(filtered, ground_truth, class_names, (0.5, 0.7), "bev"),
        }

    result = {
        "backend": "GBlobs",
        "checkpoint": str(args.checkpoint),
        "samples": len(samples),
        "classes": class_names,
        "threshold_results": threshold_results,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        threshold: {
            "3d_05_R40": values["3d"]["mAP_0.5_R40"],
            "3d_07_R40": values["3d"]["mAP_0.7_R40"],
            "bev_05_R40": values["bev"]["mAP_0.5_R40"],
            "n_gt": values["3d"]["n_gt"],
            "n_pred": values["3d"]["n_pred"],
        }
        for threshold, values in threshold_results.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
