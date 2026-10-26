#!/usr/bin/env python3
"""Masked conditional DDIM inpainting with exact noisy-ray consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-data", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--threshold", type=float, default=.50)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    files = sorted(a.seed_data.glob("*.npz"))
    if len(files) != 50:
        raise SystemExit(f"expected 50 seeds, got {len(files)}")
    device = torch.device(a.device)
    model = UNet2DModel.from_pretrained(a.checkpoint).to(device).eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.set_timesteps(a.steps, device=device)
    angles = np.load(a.corpus / "ring_angles.npy").astype(np.float32)
    az = ((np.arange(1024) + .5) / 1024 * 2 * np.pi - np.pi)[None]
    el = angles[:, None]
    direction = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.broadcast_to(np.sin(el), (64, 1024))], -1).astype(np.float32)
    generator = torch.Generator(device=device).manual_seed(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    counts = []
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for index, path in enumerate(files, 1):
            with np.load(path) as item:
                log = item["log_range"].astype(np.float32)
                observed = item["mask"].astype(bool)
            base = torch.from_numpy(np.stack([(log * 2 - 1) * observed, observed.astype(np.float32)])[None]).to(device)
            known = torch.from_numpy(np.broadcast_to(observed, (2, 64, 1024))[None]).to(device)
            fixed_noise = torch.randn(base.shape, device=device, dtype=base.dtype, generator=generator)
            x = torch.randn(base.shape, device=device, dtype=base.dtype, generator=generator)
            for step_index, timestep in enumerate(scheduler.timesteps):
                eps = model(torch.cat([x, base], 1), timestep).sample
                x = scheduler.step(eps, timestep, x).prev_sample
                if step_index + 1 < len(scheduler.timesteps):
                    next_t = scheduler.timesteps[step_index + 1].expand(1)
                    constrained = scheduler.add_noise(base, fixed_noise, next_t)
                else:
                    constrained = base
                x = torch.where(known, constrained, x)
            depth = ((x[0, 0].float().cpu().numpy() + 1) / 2).clip(0, 1)
            valid = torch.sigmoid(x[0, 1]).float().cpu().numpy() > a.threshold
            ranges = np.power(2.0, depth * 8.0) - 1.0
            points = (ranges[..., None] * direction)[valid].astype(np.float32)
            points.tofile(a.output / f"{index:04d}.bin")
            counts.append(int(len(points)))
    (a.output / "manifest.json").write_text(json.dumps({"method": "masked conditional M1 DDIM inpainting", "checkpoint": str(a.checkpoint), "steps": a.steps, "threshold": a.threshold, "point_counts": counts}, indent=2) + "\n")
    print(json.dumps({"frames": len(files), "median_points": int(np.median(counts)), "total_points": int(sum(counts))}, indent=2))


if __name__ == "__main__":
    main()
