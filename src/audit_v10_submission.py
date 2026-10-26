#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "competition/v2x_real_starting_kit_authoritative/scoring_program"
sys.path.insert(0, str(SCORING))
from detection_eval import evaluate_3d_detection  # noqa: E402


CLASSES = ("vehicle", "pedestrian", "truck")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def compact(predictions: dict, ground_truth: dict) -> dict:
    out = {}
    for iou_type in ("3d", "bev"):
        result = evaluate_3d_detection(
            predictions, ground_truth, list(CLASSES), (0.5, 0.7), iou_type
        )
        out[iou_type] = {
            "mAP_0.5_R40": result["mAP_0.5_R40"],
            "mAP_0.7_R40": result["mAP_0.7_R40"],
            "n_gt": result["n_gt"],
            "n_pred": result["n_pred"],
            "per_class_0.5_R40": {
                name: result["per_class"][name]["AP_0.5_R40"] for name in CLASSES
            },
        }
    return out


def main() -> None:
    raw_path = ROOT / "outputs/v10_authoritative/predictions_raw.json"
    zip_path = ROOT / "submissions/submission_v10_authoritative_kit_from_scratch.zip"
    val_path = ROOT / "outputs/public_val_m1_allowed_gblobs_epoch20_50_t01.json"
    ids_path = ROOT / "competition/v2x_real_starting_kit_authoritative/detection_test_frame_ids.json"
    frames_dir = ROOT / "competition/v2x_real_starting_kit_authoritative/detection_test_frames"

    raw_bytes = raw_path.read_bytes()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        pred_name = next(name for name in names if name.endswith("predictions.json"))
        zip_bytes = archive.read(pred_name)
    predictions = json.loads(zip_bytes)
    ids_obj = json.loads(ids_path.read_text())
    canonical_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    canonical_ids = [str(value) for value in canonical_ids]

    manifests = []
    for frame_id in canonical_ids:
        source = frames_dir / f"{frame_id}.bin"
        if not source.exists():
            source = frames_dir / f"{frame_id}.npy"
        manifests.append({
            "frame_id": frame_id,
            "source_path": str(source),
            "source_stem": source.stem,
            "source_sha256": sha256_file(source),
            "prediction_count": len(predictions[frame_id]),
        })

    class_stats = {}
    for name in CLASSES:
        rows = [row for values in predictions.values() for row in values if row["class"] == name]
        dims = [row["box"][3:6] for row in rows]
        class_stats[name] = {
            "count": len(rows),
            "score_min": min(row["score"] for row in rows) if rows else None,
            "mean_dimensions": [sum(x[i] for x in dims) / len(dims) for i in range(3)] if dims else None,
        }

    val = json.loads(val_path.read_text())
    val_predictions = {
        frame_id: [row for row in rows if row["score"] >= 0.1]
        for frame_id, rows in val["predictions"].items()
    }
    ground_truth = val["ground_truth"]

    roundtrip_bytes = json.dumps(val_predictions, indent=2).encode()
    roundtrip_predictions = json.loads(roundtrip_bytes)
    identity = compact(roundtrip_predictions, ground_truth)

    oracle = {
        frame_id: [
            {"box": row["box"], "class": row["class"], "score": 1.0}
            for row in rows
        ]
        for frame_id, rows in ground_truth.items()
    }
    oracle_roundtrip = json.loads(json.dumps(oracle))
    oracle_result = compact(oracle_roundtrip, ground_truth)

    permutations = []
    for target_order in itertools.permutations(CLASSES):
        mapping = dict(zip(CLASSES, target_order))
        remapped = {
            frame_id: [dict(row, **{"class": mapping[row["class"]]}) for row in rows]
            for frame_id, rows in roundtrip_predictions.items()
        }
        result = compact(remapped, ground_truth)
        permutations.append({
            "mapping": mapping,
            "3d_mAP_0.5_R40": result["3d"]["mAP_0.5_R40"],
            "bev_mAP_0.5_R40": result["bev"]["mAP_0.5_R40"],
        })

    report = {
        "package": {
            "zip_path": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
            "prediction_member": pred_name,
            "raw_sha256": sha256_bytes(raw_bytes),
            "zipped_prediction_sha256": sha256_bytes(zip_bytes),
            "byte_identical": raw_bytes == zip_bytes,
            "canonical_key_order_equal": list(predictions) == canonical_ids,
            "canonical_key_set_equal": set(predictions) == set(canonical_ids),
            "all_source_stems_equal_ids": all(row["source_stem"] == row["frame_id"] for row in manifests),
        },
        "class_stats": class_stats,
        "validation_json_roundtrip_equal": roundtrip_predictions == val_predictions,
        "validation_identity": identity,
        "validation_gt_oracle": oracle_result,
        "validation_class_permutations": sorted(
            permutations, key=lambda row: row["3d_mAP_0.5_R40"], reverse=True
        ),
        "frame_manifest": manifests,
    }
    output = ROOT / "outputs/v10_authoritative/audit_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frame_manifest"}, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
