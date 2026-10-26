#!/usr/bin/env python3
"""Render 50 synthetic XYZ clouds from static-map conditions with the learned U-Net."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_static_to_real_renderer import RendererUNet

LOW = np.asarray([-40.0, -40.0, -8.0], np.float32)
HIGH = np.asarray([40.0, 40.0, 2.0], np.float32)


def input_tensor(path: Path) -> torch.Tensor:
    with np.load(path) as d:
        log = torch.from_numpy(d["static_log"].copy()).float()
        mask = torch.from_numpy(d["static_mask"].copy()).float()
    h, w = log.shape
    yy = torch.linspace(-1, 1, h)[:, None].expand(h, w)
    xx = torch.linspace(-1, 1, w)[None, :].expand(h, w)
    return torch.stack((log * 2 - 1, mask, yy, xx))


def points_from_prediction(depth: np.ndarray, logits: np.ndarray, points: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    h, w = depth.shape
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    # Keeping the most likely returns is a learned sensor dropout simulation;
    # a small Gumbel perturbation stops repeated static maps producing clones.
    rank = np.log(np.maximum(prob, 1e-6)) + rng.gumbel(scale=.03, size=prob.shape)
    flat = np.argpartition(rank.reshape(-1), -points)[-points:]
    iy, ix = np.unravel_index(flat, (h, w))
    r = np.power(2.0, np.clip(depth[iy, ix], 0, 1) * 8.0) - 1.0
    az = ((ix.astype(np.float32) + .5) / w * 2 * np.pi - np.pi)
    el = ((iy.astype(np.float32) + .5) / h * .70 - .35)
    xyz = np.stack((r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)), 1)
    if noise > 0:
        unit = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-6)
        xyz += unit * rng.normal(0, noise, (len(xyz), 1)).astype(np.float32)
    xyz = np.clip(xyz, LOW, HIGH).astype(np.float32)
    return xyz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--points", type=int, default=9000)
    p.add_argument("--noise-m", type=float, default=.025)
    p.add_argument("--seed", type=int, default=20260806)
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location="cpu")
    model = RendererUNet(int(state["width"]))
    model.load_state_dict(state["model"])
    model.cuda().eval()
    files = sorted((a.data / "train").glob("*.npz")) + sorted((a.data / "validation").glob("*.npz"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, a.frames, dtype=int)]
    a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i, path in enumerate(files, 1):
            pred = model(input_tensor(path)[None].cuda())[0].float().cpu().numpy()
            xyz = points_from_prediction(1 / (1 + np.exp(-np.clip(pred[0], -20, 20))), pred[1], a.points, a.noise_m, rng)
            xyz.tofile(a.output / f"{i:04d}.bin")
            rows.append({"frame": f"{i:04d}.bin", "condition": str(path), "points": len(xyz), "mask_probability_mean": float((1 / (1 + np.exp(-np.clip(pred[1], -20, 20)))).mean())})
    report = {"method": "learned conditional static-map-to-real-return renderer", "policy": "only static conditioning maps used at inference; target return fields in pair archives are never loaded", "checkpoint": str(a.checkpoint), "frames": rows, "points": a.points, "seed": a.seed, "noise_m": a.noise_m}
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
