#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
V2X_ROOT = ROOT / "third_party/V2X-Real"
sys.path.insert(0, str(V2X_ROOT))
from opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center  # noqa: E402
from opencood.utils.transformation_utils import x_to_world  # noqa: E402


def metadata_path(source: Path) -> Path:
    parts = source.parts
    split_index = parts.index("train_lidar128_selective")
    split = parts[split_index + 1]
    scenario, agent, filename = parts[-3:]
    root = ROOT / "data/v2xreal_public/train_metadata_yaml"
    return root / f"{split}_agent_m1_yaml" / scenario / agent / f"{Path(filename).stem}.yaml"


def transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return boxes.copy()
    corners = boxes_to_corners_3d(boxes, order="lwh")
    homogeneous = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], axis=2)
    transformed = (matrix @ homogeneous.transpose(0, 2, 1)).transpose(0, 2, 1)[..., :3]
    return corner_to_center(transformed, order="lwh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    matches = json.loads(args.matches.read_text())
    output = {}
    manifest = []
    max_roundtrip = 0.0

    for frame_id, rows in predictions.items():
        candidates = matches[frame_id]
        best_score, best_path = max(candidates, key=lambda row: row[0])
        ranked_scores = sorted((row[0] for row in candidates), reverse=True)
        second_score = ranked_scores[1] if len(ranked_scores) > 1 else -1
        if best_score < 8 or second_score >= best_score:
            raise ValueError(f"{frame_id}: ambiguous source match {candidates[:3]}")
        source = ROOT / best_path
        meta_path = metadata_path(source)
        metadata = yaml.safe_load(meta_path.read_text())
        pose = metadata["lidar_pose"]
        matrix = x_to_world(pose)

        boxes = np.asarray([row["box"] for row in rows], dtype=np.float64)
        world = transform_boxes(boxes, matrix)
        recovered = transform_boxes(world, np.linalg.inv(matrix))
        error = float(np.max(np.abs(recovered[:, :6] - boxes[:, :6]))) if len(boxes) else 0.0
        max_roundtrip = max(max_roundtrip, error)
        output[frame_id] = [dict(row, box=[float(x) for x in box]) for row, box in zip(rows, world)]
        manifest.append({
            "frame_id": frame_id,
            "match_score": best_score,
            "second_score": second_score,
            "source": str(source),
            "metadata": str(meta_path),
            "lidar_pose": pose,
            "detections": len(rows),
            "roundtrip_max_abs_xyz_dims": error,
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.manifest.write_text(json.dumps({
        "coordinate_transform": "released -1 LiDAR local to CARLA world",
        "method": "box corners -> x_to_world(lidar_pose) -> corner_to_center(order=lwh)",
        "max_roundtrip_abs_xyz_dims": max_roundtrip,
        "frames": manifest,
    }, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.base_zip) as source_zip, zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target_zip:
        for info in source_zip.infolist():
            if info.filename != "predictions.json":
                target_zip.writestr(info, source_zip.read(info.filename))
        target_zip.write(args.output_json, "predictions.json")
    print(json.dumps({
        "frames": len(output),
        "detections": sum(map(len, output.values())),
        "max_roundtrip_abs_xyz_dims": max_roundtrip,
        "output_json": str(args.output_json),
        "output_zip": str(args.output_zip),
    }, indent=2))


if __name__ == "__main__":
    main()
