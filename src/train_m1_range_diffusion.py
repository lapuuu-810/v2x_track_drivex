#!/usr/bin/env python3
"""Train and sample an unconditional M1-native LiDAR range diffusion model."""
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class Ranges(Dataset):
    def __init__(self, directory: Path, augment: bool) -> None:
        self.files = sorted(directory.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(directory)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with np.load(self.files[index]) as item:
            depth = torch.from_numpy(item["log_range"].astype(np.float32))[None] * 2 - 1
            mask = torch.from_numpy(item["mask"].astype(np.float32))[None]
        image = torch.cat([depth, mask], 0)
        if self.augment:
            shift = int(torch.randint(0, image.shape[-1], ()).item())
            image = torch.roll(image, shift, dims=-1)
        return image


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--sample-frames", type=int, default=50)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260728)
    return p.parse_args()


def sample(model: UNet2DModel, data: Path, output: Path, frames: int, steps: int, seed: int, device: torch.device) -> dict:
    angles = np.load(data / "ring_angles.npy").astype(np.float32)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    directory = output / "synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    counts = []
    model.eval()
    with torch.inference_mode():
        for index in range(frames):
            x = torch.randn((1, 2, 64, 1024), device=device, generator=generator)
            for timestep in scheduler.timesteps:
                x = scheduler.step(model(x, timestep).sample, timestep, x).prev_sample
            depth = torch.clamp((x[0, 0] + 1) / 2, 0, 1).cpu().numpy()
            mask = (x[0, 1].cpu().numpy() > 0.5)
            ranges = (np.power(2.0, depth * 8.0) - 1.0) * mask
            azimuth = ((np.arange(1024) + 0.5) / 1024 * 2 * np.pi - np.pi)[None]
            elevation = angles[:, None]
            xyz = np.stack([
                ranges * np.cos(elevation) * np.cos(azimuth),
                ranges * np.cos(elevation) * np.sin(azimuth),
                np.broadcast_to(ranges * np.sin(elevation), ranges.shape),
            ], axis=-1)
            points = xyz[mask].astype(np.float32)
            points.tofile(directory / f"{index + 1:04d}.bin")
            counts.append(int(len(points)))
    return {"frames": frames, "median_points": int(np.median(counts)), "total_points": int(sum(counts))}


def main() -> None:
    cfg = parse()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if distributed else 0
    torch.manual_seed(cfg.seed + rank)
    dataset = Ranges(cfg.data / "train", augment=True)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0, drop_last=True)
    model = UNet2DModel(sample_size=(64, 1024), in_channels=2, out_channels=2, layers_per_block=2,
                        block_out_channels=(48, 96, 128, 128), down_block_types=("DownBlock2D",) * 4,
                        up_block_types=("UpBlock2D",) * 4, norm_num_groups=8).to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=False)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    cfg.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        wrapped.train()
        total = torch.zeros(2, device=device)
        for image in loader:
            image = image.to(device, non_blocking=True)
            noise = torch.randn_like(image)
            time = torch.randint(0, scheduler.config.num_train_timesteps, (image.shape[0],), device=device, dtype=torch.long)
            noisy = scheduler.add_noise(image, noise, time)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = F.mse_loss(wrapped(noisy, time).sample.float(), noise.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            total += torch.tensor([loss.item(), 1], device=device)
        if distributed:
            dist.all_reduce(total)
        if rank == 0:
            row = {"epoch": epoch, "loss": float(total[0] / total[1])}
            history.append(row); print(json.dumps(row), flush=True)
            (cfg.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                raw = wrapped.module if distributed else wrapped
                raw.save_pretrained(cfg.output / f"epoch_{epoch:03d}", safe_serialization=True)
    if distributed:
        dist.barrier()
    if rank == 0:
        raw = wrapped.module if distributed else wrapped
        report = sample(raw, cfg.data, cfg.output, cfg.sample_frames, cfg.sample_steps, cfg.seed + 99, device)
        report.update({"method": "unconditional M1-native 64x1024 range diffusion", "epochs": cfg.epochs, "sample_steps": cfg.sample_steps})
        (cfg.output / "generation_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
