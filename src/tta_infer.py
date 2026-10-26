#!/usr/bin/env python3
"""TTA inference for the fixed 50-frame local protocol (and test frames).

Runs the model on geometric transforms of each frame (flips / 90-degree
rotations), transforms the boxes back, then fuses all variants with
class-wise NMS. Output format matches final_submission_prepare caches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402
if os.environ.get("GBLOBS") == "1":
    sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "GBlobs" / "tools"),
                    str(PROJECT_ROOT / "third_party" / "GBlobs"),
                    str(PROJECT_ROOT / "third_party" / "V2X-Real"),
                    str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"),
                    str(PROJECT_ROOT / "src")]
else:
    sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "OpenPCDet" / "tools"),
                    str(PROJECT_ROOT / "third_party" / "OpenPCDet"),
                    str(PROJECT_ROOT / "third_party" / "V2X-Real"),
                    str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program"),
                    str(PROJECT_ROOT / "src")]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from common import read_points  # noqa: E402

CLASSES = ["vehicle", "pedestrian", "truck"]


def apply_transform(points, t):
    p = points.copy()
    if t == "flip_x":
        p[:, 0] *= -1.0
    elif t == "flip_y":
        p[:, 1] *= -1.0
    elif t == "flip_xy":
        p[:, 0] *= -1.0
        p[:, 1] *= -1.0
    elif t == "rot90":
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = -y
        p[:, 1] = x
    elif t == "rot180":
        p[:, 0] *= -1.0
        p[:, 1] *= -1.0
    elif t == "rot270":
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = y
        p[:, 1] = -x
    return p


def untransform_box(box, t):
    x, y, z, w, d, h, th = box[:7]
    if t == "flip_x":
        # Model canonicalizes heading to the undirected range after flip_x:
        # observed output is (pi - th), so invert with (pi - th).
        th = (math.pi - th + math.pi) % (2 * math.pi) - math.pi
        return [-x, y, z, w, d, h, th]
    if t == "flip_y":
        return [x, -y, z, w, d, h, -th]
    if t == "flip_xy":
        return [-x, -y, z, w, d, h, th + math.pi]
    if t == "rot90":
        return [-y, x, z, w, d, h, th + math.pi / 2]
    if t == "rot180":
        return [-x, -y, z, w, d, h, th + math.pi]
    if t == "rot270":
        return [y, -x, z, w, d, h, th - math.pi / 2]
    return list(box[:7])


def iou3d(a, b):
    ax1, ay1, az1 = a[:3] - a[3:6] / 2
    ax2, ay2, az2 = a[:3] + a[3:6] / 2
    bx1, by1, bz1 = b[:3] - b[3:6] / 2
    bx2, by2, bz2 = b[:3] + b[3:6] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iy * iz
    va = float(np.prod(a[3:6]))
    vb = float(np.prod(b[3:6]))
    return inter / max(1e-9, va + vb - inter)


def box_avg_fuse(variant_rows, iou_threshold=0.5, score_floor=0.001):
    """Group near-duplicate boxes across transforms and average them
    (score-weighted), which reduces center/size noise vs keep-max NMS."""
    out = []
    for cls in CLASSES:
        rows = sorted([r for r in variant_rows
                       if r["class"] == cls and r["score"] >= score_floor],
                      key=lambda r: r["score"], reverse=True)
        groups = []
        for r in rows:
            b = np.asarray(r["box"][:7], dtype=float)
            placed = False
            for g in groups:
                if iou3d(b, g["box"]) >= iou_threshold:
                    w = r["score"]
                    g["box"] = (g["box"] * g["w"] + b[:7] * w) / (g["w"] + w)
                    g["w"] += w
                    g["score"] = max(g["score"], r["score"])
                    placed = True
                    break
            if not placed:
                groups.append({"box": b[:7].copy(), "score": r["score"], "w": r["score"]})
        for g in groups:
            h = float(g["box"][5])
            th = g["box"][6]
            # normalize heading to [-pi, pi]
            th = (th + np.pi) % (2 * np.pi) - np.pi
            g["box"][6] = th
            out.append({"box": [float(x) for x in g["box"][:7]],
                        "score": float(g["score"]), "class": cls})
    return out


def fuse_variants(variant_rows, nms_threshold=0.2, score_floor=0.001):
    out = []
    for cls in CLASSES:
        rows = sorted([r for r in variant_rows
                       if r["class"] == cls and r["score"] >= score_floor],
                      key=lambda r: r["score"], reverse=True)
        keep = []
        for r in rows:
            box = np.asarray(r["box"][:7], dtype=float)
            if all(iou3d(box, np.asarray(k["box"][:7], dtype=float)) < nms_threshold
                   for k in keep):
                keep.append(r)
        out.extend(keep)
    return out


class TTADataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, samples, transform, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=Path("."), logger=logger)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        points = read_points(sample["bin"], sample.get("width"))
        if points.shape[1] > 3:
            points[:, 3] = 0.0
        points = apply_transform(points, self.transform)
        return self.prepare_data({"points": points, "frame_id": sample["id"]})


def ground_truth_for_sample(sample):
    import copy
    import yaml
    from opencood.data_utils import SUPER_CLASS_MAP
    from opencood.utils import box_utils

    inv = {fine: sup for sup, fines in SUPER_CLASS_MAP.items() for fine in fines}
    PC_RANGE = [-40, -40, -8, 40, 40, 2]
    meta = yaml.safe_load(sample["yaml"].read_text(encoding="utf-8"))
    grouped = {}
    for oid, obj in meta.get("vehicles", {}).items():
        sup = inv.get(obj.get("obj_type"))
        if sup is None:
            continue
        o = copy.deepcopy(obj)
        o["obj_type"] = np.asarray([sup])
        grouped[oid] = o
    projected = {}
    box_utils.project_world_objects(grouped, projected, meta["lidar_pose"],
                                    PC_RANGE, "lwh")
    rows = []
    for box_and_class in projected.values():
        row = box_and_class[0]
        rows.append({"box": [float(x) for x in row[:7]], "class": str(row[7]).lower()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--frame-ids", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--transforms", nargs="+",
                    default=["id", "flip_x", "flip_y", "flip_xy"])
    ap.add_argument("--fuse-nms", type=float, default=0.2)
    ap.add_argument("--fuse-mode", choices=["nms", "avg"], default="nms")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    transforms = list(args.transforms)
    samples = []
    if args.manifest is not None:
        for m in json.loads(args.manifest.read_text()):
            bin_path = PROJECT_ROOT / m["path"]
            yaml_path = bin_path.with_suffix(".yaml")
            samples.append({"id": m["id"], "bin": bin_path, "yaml": yaml_path,
                            "width": m.get("width", 4)})
    elif args.frames is not None and args.frame_ids is not None:
        ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
        fids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
        fids = [str(x) for x in fids]
        files = sorted([x for x in args.frames.iterdir() if x.suffix.lower() in {".bin", ".npy"}])
        by_stem = {x.stem: x for x in files}
        paths = [by_stem[x] for x in fids]
        samples = [{"id": fid, "bin": pth, "yaml": None, "width": 3}
                   for fid, pth in zip(fids, paths)]
    else:
        raise SystemExit("need --manifest or --frames/--frame-ids")

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.config), cfg)
    if cfg.MODEL.DENSE_HEAD.NAME == "CenterHead":
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = 0.001
    else:
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]

    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), TTADataset(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, samples, "id", logger))
    model.load_params_from_file(str(args.checkpoint), logger=logger, to_cpu=True)
    torch.cuda.set_device(args.device)
    model.cuda().eval()

    predictions, ground_truth = {}, {}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            variant_rows = []
            for t in transforms:
                dataset = TTADataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, [sample],
                                    t, logger)
                batch = dataset.collate_batch([dataset[0]])
                load_data_to_gpu(batch)
                pred = model.forward(batch)[0][0]
                boxes = pred["pred_boxes"].cpu().numpy()
                scores = pred["pred_scores"].cpu().numpy()
                labels = pred["pred_labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    ub = untransform_box(box[:7], t)
                    variant_rows.append({
                        "box": [float(x) for x in ub],
                        "score": float(score),
                        "class": cfg.CLASS_NAMES[int(label) - 1],
                    })
            if args.fuse_mode == "avg":
                predictions[sample["id"]] = box_avg_fuse(variant_rows, 0.5, 0.001)
            else:
                predictions[sample["id"]] = fuse_variants(variant_rows,
                                                          args.fuse_nms, 0.001)
            if sample.get("yaml") is not None and sample["yaml"].exists():
                ground_truth[sample["id"]] = ground_truth_for_sample(sample)
            else:
                ground_truth[sample["id"]] = []
            if (index + 1) % 10 == 0:
                logger.info("tta %d/%d", index + 1, len(samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "transforms": transforms,
        "samples": len(samples),
        "predictions": predictions,
        "ground_truth": ground_truth,
    }, indent=1), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": len(predictions),
                      "boxes": sum(map(len, predictions.values()))}))


if __name__ == "__main__":
    main()
