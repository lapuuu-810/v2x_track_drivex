#!/usr/bin/env python3
"""Assemble the v17 (Combined 0.7421) final predictions and submission archive.

The fusion procedure is preserved exactly:
  vehicle   = nms_fuse(union(B4,B7,B10,B11 vehicle rows), IoU 0.5, floor 0.03)
  truck     = nms_fuse(union(d14,d50k truck rows),        IoU 0.2, floor 0.03)
  pedestrian= B4 pedestrian rows
  rows      = blend(vehicle, 0.2) + blend(truck, 0.2) + blend(pedestrian, 0.25)
  final     = nms_fuse(rows, IoU 0.2, floor 0.03)

Center blend (per box): rotate points into the box frame, keep points with
|x|<=dx/2*1.4, |y|<=dy/2*1.4, |z-zc|<=max(dz,1.0); if >=3 points, move the box
center toward the point centroid: c' = (1-alpha)c + alpha*centroid (x,y,z).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def nms_fuse(rows, thr, floor):
    out = []
    for cls in ("vehicle", "pedestrian", "truck"):
        rc = sorted(
            [r for r in rows if r["class"] == cls and r["score"] >= floor],
            key=lambda r: -r["score"],
        )
        keep = []
        for r in rc:
            b = np.asarray(r["box"], dtype=float)
            if all(iou3d(b, np.asarray(k["box"], dtype=float)) < thr for k in keep):
                keep.append(r)
        out.extend(keep)
    return out


def blend_rows(rows, fid, pts, blend):
    out = []
    for r in rows:
        b = list(r["box"])
        if len(pts) == 0:
            out.append(r)
            continue
        x, y, z, dx, dy, dz, th = b
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        local = (pts[:, :2] - np.array([x, y])) @ R
        m = (
            (np.abs(local[:, 0]) <= dx / 2 * 1.4)
            & (np.abs(local[:, 1]) <= dy / 2 * 1.4)
            & (np.abs(pts[:, 2] - z) <= max(dz, 1.0))
        )
        sel = pts[m]
        if len(sel) < 3:
            out.append(r)
            continue
        cx, cy, cz = sel[:, :3].mean(axis=0)
        nb = list(b)
        nb[0] = x + blend * (cx - x)
        nb[1] = y + blend * (cy - y)
        nb[2] = z + blend * (cz - z)
        out.append({"box": nb, "score": r["score"], "class": r["class"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frames", type=Path, required=True, help="official test frames dir")
    ap.add_argument("--frame-ids", type=Path, required=True)
    ap.add_argument("--b4", type=Path, required=True)
    ap.add_argument("--b7", type=Path, required=True)
    ap.add_argument("--b10", type=Path, required=True)
    ap.add_argument("--b11", type=Path, required=True)
    ap.add_argument("--d14", type=Path, required=True)
    ap.add_argument("--d50", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True, help="regenerated S12-x092 frames dir")
    ap.add_argument("--declaration", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from common import read_points  # noqa: E402

    b4 = load_json(args.b4)
    b7 = load_json(args.b7)
    b10 = load_json(args.b10)
    b11 = load_json(args.b11)
    d14 = load_json(args.d14)
    d50 = load_json(args.d50)
    fids = [str(x) for x in load_json(args.frame_ids)]
    ids_obj = load_json(args.frame_ids)
    fids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    fids = [str(x) for x in fids]

    pts_cache = {}

    def get_pts(fid):
        if fid not in pts_cache:
            p = args.test_frames / f"{fid}.bin"
            pts_cache[fid] = read_points(p, 3) if p.exists() else np.zeros((0, 4), dtype=np.float32)
        return pts_cache[fid]

    preds = {}
    for fid in fids:
        veh_rows = (
            [r for r in b4.get(fid, []) if r["class"] == "vehicle"]
            + [r for r in b7.get(fid, []) if r["class"] == "vehicle"]
            + [r for r in b10.get(fid, []) if r["class"] == "vehicle"]
            + [r for r in b11.get(fid, []) if r["class"] == "vehicle"]
        )
        veh = nms_fuse(veh_rows, 0.5, 0.03)
        trk = nms_fuse(
            [r for r in d14.get(fid, []) + d50.get(fid, []) if r["class"] == "truck"],
            0.2, 0.03,
        )
        ped = [r for r in b4.get(fid, []) if r["class"] == "pedestrian"]
        rows = (
            blend_rows(veh, fid, get_pts(fid), 0.2)
            + blend_rows(trk, fid, get_pts(fid), 0.2)
            + blend_rows(ped, fid, get_pts(fid), 0.25)
        )
        preds[fid] = nms_fuse(rows, 0.2, 0.03)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = args.output_dir / "predictions_final_v17.json"
    pred_out.write_text(json.dumps(preds, indent=1), encoding="utf-8")
    n_pred = sum(len(v) for v in preds.values())
    print(f"frames: {len(preds)}  boxes: {n_pred}  -> {pred_out}")

    # package: predictions + synthetic (S12-x092) + declaration
    zip_out = args.output_dir / "submission_final_v17_repro.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.json", json.dumps(preds))
        frames = sorted(args.synthetic.glob("*.bin")) + sorted(args.synthetic.glob("*.npy"))
        for p in frames:
            z.write(p, f"synthetic/{p.name}")
        z.write(args.declaration, "declaration.txt")
    sha = hashlib.sha256(zip_out.read_bytes()).hexdigest()
    print(f"wrote {zip_out}  sha256: {sha}")
    print(f"predictions sha256: {hashlib.sha256(pred_out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
