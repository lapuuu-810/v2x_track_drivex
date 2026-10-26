#!/usr/bin/env python3
"""Cache public V2X-Real predictions and GT without running slow exact AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval_v2xreal_public import (
    PublicICDataset,
    apply_profile,
    build_network,
    cfg,
    cfg_from_yaml_file,
    common_utils,
    discover_ic_samples,
    ground_truth_for_sample,
    load_data_to_gpu,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("car", "three_class"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.config), cfg)
    apply_profile(args.profile)
    samples = discover_ic_samples(args.data_root, args.stride, args.max_frames)
    logger.info("Caching %d public IC samples", len(samples))
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
                logger.info("Cached %d/%d", index + 1, len(samples))

    result = {
        "profile": args.profile,
        "checkpoint": str(args.checkpoint),
        "samples": len(samples),
        "classes": list(cfg.CLASS_NAMES),
        "predictions": predictions,
        "ground_truth": ground_truth,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding="utf-8")
    print(f"Wrote {args.output} with {sum(map(len, predictions.values()))} predictions")


if __name__ == "__main__":
    main()
