#!/usr/bin/env python3
"""Evaluate a synthetic-only OpenPCDet model on public V2X-Real IC data.

The public labels are used only for validation and threshold selection. They
are never converted into training samples or passed to the detector.
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
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"
V2XREAL_ROOT = PROJECT_ROOT / "third_party" / "V2X-Real"
SCORING_ROOT = PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"
sys.path[:0] = [
    str(OPENPCDET_ROOT / "tools"),
    str(OPENPCDET_ROOT),
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
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=Path("."),
            logger=logger,
        )
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        points = np.fromfile(sample["bin"], dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        # The challenge frames contain XYZ only. Keeping the public raw return
        # counts as intensity would introduce an avoidable domain mismatch.
        points[:, 3] = 0.0
        return self.prepare_data({"points": points, "frame_id": sample["id"]})


def discover_ic_samples(root: Path, stride: int, max_frames: int | None, forbidden: set[str] | None = None):
    """Match official IC validation behavior: the first negative agent is ego."""
    samples = []
    for scenario_index, scenario in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        infra_agents = sorted(
            (p for p in scenario.iterdir() if p.is_dir() and int(p.name) < 0),
            key=lambda p: p.name,
        )
        if not infra_agents:
            continue
        ego = infra_agents[0]
        yamls = sorted(p for p in ego.glob("*.yaml") if "additional" not in p.name)
        for yaml_path in yamls[::stride]:
            if forbidden and yaml_path.stem in forbidden:
                continue
            bin_path = yaml_path.with_suffix(".bin")
            if not bin_path.exists():
                continue
            samples.append({
                "id": f"{scenario_index}_{yaml_path.stem}",
                "scenario": scenario.name,
                "agent": ego.name,
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
    box_utils.project_world_objects(
        grouped,
        projected,
        metadata["lidar_pose"],
        PC_RANGE,
        "lwh",
    )
    rows = []
    for box_and_class in projected.values():
        row = box_and_class[0]
        rows.append({
            "box": [float(x) for x in row[:7]],
            "class": str(row[7]).lower(),
        })
    return rows


def apply_profile(profile: str):
    # DSVT uses CenterHead and its checkpoint is trained jointly on all 3 classes.
    # Do not overwrite backbone dimensions or inject anchor-head settings here.
    if cfg.MODEL.DENSE_HEAD.NAME == "CenterHead":
        if profile != "three_class":
            raise ValueError("CenterHead validation requires --profile three_class")
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = 0.001
        return

    cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES = 384
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.001
    cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS = False
    cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = 0.01
    if profile == "car":
        cfg.CLASS_NAMES = ["vehicle"]
        cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = [{
            "class_name": "vehicle",
            "anchor_sizes": [[4.67, 2.01, 1.52]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-3.04],
            "align_center": True,
            "feature_map_stride": 8,
            "matched_threshold": 0.55,
            "unmatched_threshold": 0.4,
        }]
    elif profile == "three_class":
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]
        cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = [
            {
                "class_name": "vehicle",
                "anchor_sizes": [[4.67, 2.01, 1.52]],
                "anchor_rotations": [0, 1.57],
                "anchor_bottom_heights": [-3.04],
                "align_center": True,
                "feature_map_stride": 8,
                "matched_threshold": 0.55,
                "unmatched_threshold": 0.4,
            },
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
    else:
        raise ValueError(profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("car", "three_class"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--forbidden", type=Path)
    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.config), cfg)
    apply_profile(args.profile)
    forbidden = {line.strip().split("_")[-1] for line in args.forbidden.read_text().splitlines() if line.strip() and not line.startswith("#")} if args.forbidden else set()
    samples = discover_ic_samples(args.data_root, args.stride, args.max_frames, forbidden)
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
                    "box": [float(x) for x in box[:7]],
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
            frame_id: [p for p in rows if p["score"] >= threshold]
            for frame_id, rows in predictions.items()
        }
        threshold_results[str(threshold)] = {
            "3d": evaluate_3d_detection(filtered, ground_truth, class_names, (0.5, 0.7), "3d"),
            "bev": evaluate_3d_detection(filtered, ground_truth, class_names, (0.5, 0.7), "bev"),
        }

    result = {
        "profile": args.profile,
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
