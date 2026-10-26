#!/usr/bin/env python3
"""Conditional DDIM inpainting for native M1 range images.

The network is trained only on permitted real training scans.  A four-way
azimuth-thinned scan is the condition and the original scan is the target;
at inference the condition is a projected UrbanTwin scan.  This gives the
diffusion model a scene-geometry signal, unlike the earlier unconditional
M1 diffusion baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class RangePairs(Dataset):
    def __init__(self, directory: Path) -> None:
        self.files = sorted(directory.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(directory)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as item:
            log = torch.from_numpy(item["log_range"].astype(np.float32))
            mask = torch.from_numpy(item["mask"].astype(np.float32))
        shift = int(torch.randint(0, log.shape[-1], ()).item())
        log, mask = torch.roll(log, shift, -1), torch.roll(mask, shift, -1)
        phase = int(torch.randint(0, 4, ()).item())
        cols = torch.arange(mask.shape[-1])
        keep = ((cols + phase) % 4 == 0)[None].expand_as(mask) & (mask > 0)
        keep &= torch.rand_like(mask) > 0.08
        condition = torch.stack([(log * 2 - 1) * keep, keep.float()])
        target = torch.stack([log * 2 - 1, mask])
        return condition, target


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seed-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=20260729)
    return p


def make_model() -> UNet2DModel:
    return UNet2DModel(
        sample_size=(64, 1024), in_channels=4, out_channels=2,
        layers_per_block=2, block_out_channels=(48, 96, 128, 128),
        down_block_types=("DownBlock2D",) * 4,
        up_block_types=("UpBlock2D",) * 4, norm_num_groups=8,
    )


def sample(model: UNet2DModel, seed_data: Path, corpus: Path, output: Path, steps: int, threshold: float, seed: int, device: torch.device) -> None:
    files = sorted(seed_data.glob("*.npz"))
    if len(files) != 50:
        raise RuntimeError(f"expected 50 projected seeds, got {len(files)}")
    angles = np.load(corpus / "ring_angles.npy").astype(np.float32)
    az = ((np.arange(1024) + .5) / 1024 * 2 * np.pi - np.pi)[None]
    el = angles[:, None]
    direction = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.broadcast_to(np.sin(el), (64, 1024))], -1).astype(np.float32)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    counts = []
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for index, path in enumerate(files, 1):
            with np.load(path) as item:
                log = item["log_range"].astype(np.float32)
                mask = item["mask"].astype(np.float32)
            cond = torch.from_numpy(np.stack([(log * 2 - 1) * mask, mask])[None]).to(device)
            x = torch.randn((1, 2, 64, 1024), device=device, generator=generator)
            for timestep in scheduler.timesteps:
                eps = model(torch.cat([x, cond], 1), timestep).sample
                x = scheduler.step(eps, timestep, x).prev_sample
            depth = ((x[0, 0].float().cpu().numpy() + 1) / 2).clip(0, 1)
            valid = torch.sigmoid(x[0, 1]).float().cpu().numpy() > threshold
            ranges = np.power(2.0, depth * 8.0) - 1.0
            points = (ranges[..., None] * direction)[valid].astype(np.float32)
            points.tofile(output / f"{index:04d}.bin")
            counts.append(int(len(points)))
    (output / "manifest.json").write_text(json.dumps({"method": "conditional sparse-to-native M1 DDIM", "frames": len(files), "steps": steps, "threshold": threshold, "point_counts": counts}, indent=2) + "\n")
    print(json.dumps({"frames": len(files), "median_points": int(np.median(counts)), "total_points": int(sum(counts))}, indent=2), flush=True)


def main() -> None:
    cfg = parser().parse_args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(cfg.seed + rank)
    dataset = RangePairs(cfg.data / "train")
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)
    model = make_model().to(device)
    wrapped = DDP(model, device_ids=[rank]) if distributed else model
    noise = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=False)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    cfg.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        total = torch.zeros(2, device=device)
        wrapped.train()
        for condition, target in loader:
            condition, target = condition.to(device, non_blocking=True), target.to(device, non_blocking=True)
            epsilon = torch.randn_like(target)
            time = torch.randint(0, noise.config.num_train_timesteps, (len(target),), device=device, dtype=torch.long)
            noisy = noise.add_noise(target, epsilon, time)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = F.mse_loss(wrapped(torch.cat([noisy, condition], 1), time).sample.float(), epsilon.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += torch.tensor([loss.item(), 1], device=device)
        if distributed:
            dist.all_reduce(total)
        if not distributed or dist.get_rank() == 0:
            row = {"epoch": epoch, "loss": float(total[0] / total[1])}
            history.append(row); print(json.dumps(row), flush=True)
            (cfg.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                raw = wrapped.module if distributed else wrapped
                raw.save_pretrained(cfg.output / f"epoch_{epoch:03d}", safe_serialization=True)
    if distributed:
        dist.barrier()
    if not distributed or dist.get_rank() == 0:
        raw = wrapped.module if distributed else wrapped
        sample(raw, cfg.seed_data, cfg.data, cfg.output / "synthetic", cfg.sample_steps, cfg.threshold, cfg.seed + 99, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
