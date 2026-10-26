#!/usr/bin/env python3
"""Prepare frame-disjoint static-map -> real-return supervision from M2 train.

For every target scan, the conditioning map is built from a fixed set of
other frames in the same infrastructure scene.  The target frame is never
included in its map.  This provides legal supervision for a sensor-return
renderer: the network must learn dropout, local range bias and dynamic return
statistics rather than copying a real target scan.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LOW = np.asarray([-40.0, -40.0, -8.0], np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], np.float32)


def crop(x: np.ndarray) -> np.ndarray:
    keep = np.isfinite(x).all(1) & np.any(x != 0, 1)
    keep &= (x >= LOW).all(1) & (x <= HIGH).all(1)
    return x[keep].astype(np.float32, copy=False)


def grid(x: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if not len(x):
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    radial = np.linalg.norm(x[:, :2], axis=1)
    r = np.linalg.norm(x, axis=1)
    el = np.arctan2(x[:, 2], np.maximum(radial, 1e-6))
    az = np.arctan2(x[:, 1], x[:, 0])
    iy = np.clip(((el + .35) / .70 * h).astype(np.int32), 0, h - 1)
    ix = np.clip(((az + np.pi) / (2 * np.pi) * w).astype(np.int32), 0, w - 1)
    flat = iy.astype(np.int64) * w + ix
    order = np.argsort(r, kind="stable")
    _, first = np.unique(flat[order], return_index=True)
    selected = order[first]
    out = np.zeros((h, w), np.float32)
    out[iy[selected], ix[selected]] = r[selected]
    mask = (out > 0).astype(np.float32)
    log = np.zeros_like(out)
    log[mask > 0] = np.log2(out[mask > 0] + 1.0) / 8.0
    return log, mask


def stable_map(context: list[Path], voxel: float, min_hits: int) -> np.ndarray:
    count: Counter[tuple[int, int, int]] = Counter()
    for p in context:
        raw = np.fromfile(p, np.float32)
        x = crop(raw.reshape(-1, 4)[:, :3])
        code = np.floor((x - LOW) / voxel).astype(np.int16)
        count.update(map(tuple, np.unique(code, axis=0)))
    code = np.asarray([k for k, v in count.items() if v >= min_hits], np.int16)
    if not len(code):
        raise RuntimeError("empty static map")
    return LOW + (code.astype(np.float32) + .5) * voxel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-m2-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context-frames", type=int, default=12)
    p.add_argument("--targets-per-scene", type=int, default=20)
    p.add_argument("--voxel-m", type=float, default=.10)
    p.add_argument("--min-hit-fraction", type=float, default=.17)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--val-mod", type=int, default=5, help="scene index modulo used for validation")
    p.add_argument("--val-rem", type=int, default=0)
    a = p.parse_args()
    bad = {x.strip().split("_")[-1] for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith("#")}
    scenes = []
    for d in sorted(x for x in a.real_m2_root.rglob("-2") if x.is_dir()):
        files = [x for x in sorted(d.glob("*.bin")) if x.stem not in bad]
        if len(files) > a.context_frames + 1:
            scenes.append((str(d.parent.relative_to(a.real_m2_root)), files))
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_i, (name, files) in enumerate(scenes):
        context_i = set(np.linspace(0, len(files) - 1, min(a.context_frames, len(files)), dtype=int).tolist())
        context = [f for i, f in enumerate(files) if i in context_i]
        pool = [f for i, f in enumerate(files) if i not in context_i]
        targets = [pool[i] for i in np.linspace(0, len(pool) - 1, min(a.targets_per_scene, len(pool)), dtype=int)]
        world = stable_map(context, a.voxel_m, max(2, int(np.ceil(len(context) * a.min_hit_fraction))))
        sl, sm = grid(world, a.height, a.width)
        subset = "validation" if scene_i % a.val_mod == a.val_rem else "train"
        out = a.output / subset
        out.mkdir(exist_ok=True)
        for target in targets:
            raw = np.fromfile(target, np.float32)
            tl, tm = grid(crop(raw.reshape(-1, 4)[:, :3]), a.height, a.width)
            key = f"{scene_i:03d}_{target.stem}"
            np.savez_compressed(out / f"{key}.npz", static_log=sl, static_mask=sm, target_log=tl, target_mask=tm)
            records.append({"key": key, "subset": subset, "scene": name, "target": str(target), "context_frames": [str(x) for x in context], "map_voxels": int(len(world))})
        print(f"{scene_i + 1}/{len(scenes)} {subset} {name}: {len(world)} map voxels, {len(targets)} targets", flush=True)
    manifest = {"method": "frame-disjoint temporal static map conditioning", "policy": "target frame excluded from its map; forbidden frames excluded", "shape": [a.height, a.width], "records": records, "config": vars(a)}
    (a.output / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    print(json.dumps({"scenes": len(scenes), "train": sum(r["subset"] == "train" for r in records), "validation": sum(r["subset"] == "validation" for r in records)}, indent=2))


if __name__ == "__main__":
    main()
