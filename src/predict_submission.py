#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"
sys.path.insert(0, str(OPENPCDET_ROOT / "tools"))
sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from common import read_points  # noqa: E402


class ChallengeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, paths: list[Path], ids: list[str], binary_width: int | None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=Path("."), logger=common_utils.create_logger())
        self.paths = paths
        self.ids = ids
        self.binary_width = binary_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        points = read_points(self.paths[index], self.binary_width)
        n_feat = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if points.shape[-1] != n_feat:
            points = points[:, :n_feat].copy()
        return self.prepare_data({"points": points, "frame_id": self.ids[index]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OpenPCDet checkpoint on challenge test frames")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-width", type=int, choices=(3, 4))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()

    cfg_from_yaml_file(str(args.config), cfg)
    frame_ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = frame_ids_obj.get("frame_ids", frame_ids_obj) if isinstance(frame_ids_obj, dict) else frame_ids_obj
    frame_ids = [str(x) for x in frame_ids]
    files = sorted([p for p in args.frames.iterdir() if p.suffix.lower() in {".bin", ".npy"}])
    by_stem = {p.stem: p for p in files}
    if all(x in by_stem for x in frame_ids):
        paths = [by_stem[x] for x in frame_ids]
    elif len(files) == len(frame_ids):
        paths = files
    else:
        raise SystemExit(f"Found {len(files)} point clouds but {len(frame_ids)} canonical frame IDs")

    dataset = ChallengeDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, paths, frame_ids, args.binary_width)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.checkpoint), logger=dataset.logger, to_cpu=True)
    model.cuda().eval()

    predictions = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            pred_dicts, _ = model.forward(batch)
            pred = pred_dicts[0]
            boxes = pred["pred_boxes"].detach().cpu().numpy()
            scores = pred["pred_scores"].detach().cpu().numpy()
            labels = pred["pred_labels"].detach().cpu().numpy()
            rows = []
            for box, score, label in zip(boxes, scores, labels):
                if float(score) < args.score_threshold:
                    continue
                rows.append({
                    "box": [float(x) for x in box[:7]],
                    "score": float(np.clip(score, 0, 1)),
                    "class": str(cfg.CLASS_NAMES[int(label) - 1]).lower(),
                })
            predictions[frame_ids[index]] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote predictions for {len(predictions)} frames to {args.output}")


if __name__ == "__main__":
    main()

