#!/usr/bin/env python3
"""Self-supervised sparse-to-native M1 LiDAR range restoration.

The model never pairs UrbanTwin with a real scan.  Instead, it is trained on
official *training* scans by masking their valid rays to an UrbanTwin-like
budget and reconstructing the original scan.  At inference the same learned
sensor prior restores a projected UrbanTwin range image, preserving source
geometry while generating new M1-native returns.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class Ranges(Dataset):
    def __init__(self, root: Path, augment: bool) -> None:
        self.files = sorted(root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(root)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as item:
            log = torch.from_numpy(item["log_range"].astype(np.float32))
            mask = torch.from_numpy(item["mask"].astype(np.float32))
        if self.augment:
            shift = int(torch.randint(0, log.shape[-1], ()).item())
            log, mask = torch.roll(log, shift, -1), torch.roll(mask, shift, -1)
        # Four-way azimuth thinning gives M1's ~35k native returns an input
        # budget close to the 9k UrbanTwin candidate budget.
        phase = int(torch.randint(0, 4, ()).item())
        cols = torch.arange(mask.shape[-1])
        keep = ((cols + phase) % 4 == 0)[None].expand_as(mask) & (mask > 0)
        # Controlled random dropout prevents the restorer from encoding a
        # deterministic checkerboard assumption.
        keep &= torch.rand_like(mask) > 0.08
        source = torch.stack([(log * 2 - 1) * keep, keep.float()])
        target = torch.stack([log * 2 - 1, mask])
        return source, target


def block(inp: int, out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, out, 3, stride, 1), nn.GroupNorm(8, out), nn.SiLU())


class Restorer(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.e1 = nn.Sequential(block(2, width), block(width, width))
        self.e2 = nn.Sequential(block(width, width * 2, 2), block(width * 2, width * 2))
        self.e3 = nn.Sequential(block(width * 2, width * 3, 2), block(width * 3, width * 3))
        self.e4 = nn.Sequential(block(width * 3, width * 4, 2), block(width * 4, width * 4))
        self.u3 = nn.Sequential(nn.ConvTranspose2d(width * 4, width * 3, 4, 2, 1), nn.SiLU(), block(width * 3, width * 3))
        self.u2 = nn.Sequential(nn.ConvTranspose2d(width * 6, width * 2, 4, 2, 1), nn.SiLU(), block(width * 2, width * 2))
        self.u1 = nn.Sequential(nn.ConvTranspose2d(width * 4, width, 4, 2, 1), nn.SiLU(), block(width, width))
        self.out = nn.Conv2d(width * 2, 2, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x); e2 = self.e2(e1)
        e3 = self.e3(e2); e4 = self.e4(e3)
        u3 = self.u3(e4)
        u2 = self.u2(torch.cat([u3, e3], 1))
        u1 = self.u1(torch.cat([u2, e2], 1))
        residual = self.out(torch.cat([u1, e1], 1))
        depth = (x[:, :1] + 0.32 * torch.tanh(residual[:, :1])).clamp(-1, 1)
        return torch.cat([depth, residual[:, 1:]], 1)


class PatchDiscriminator(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2, width, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(width, width * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(width * 2, width * 3, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(width * 3, 1, 3, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def losses(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = target[:, 1:2]
    depth = (pred[:, :1] - target[:, :1]).abs()
    depth = (depth * valid).sum() / valid.sum().clamp_min(1)
    mask = F.binary_cross_entropy_with_logits(pred[:, 1:2], target[:, 1:2])
    grad = (pred[:, :1, :, 1:] - pred[:, :1, :, :-1]).abs().mean()
    return depth, mask, grad


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--init", type=Path, help="optional compatible restorer checkpoint")
    p.add_argument("--seed", type=int, default=20260729)
    return p.parse_args()


def main() -> None:
    cfg = parse(); distributed = "LOCAL_RANK" in os.environ
    if distributed: dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(rank); device = torch.device("cuda", rank)
    torch.manual_seed(cfg.seed + rank)
    train = Ranges(cfg.data / "train", augment=True)
    sampler = DistributedSampler(train, shuffle=True) if distributed else None
    loader = DataLoader(train, batch_size=cfg.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)
    g = Restorer().to(device)
    if cfg.init:
        g.load_state_dict(torch.load(cfg.init, map_location="cpu")["state_dict"])
    d = PatchDiscriminator().to(device)
    if distributed: g, d = DDP(g, device_ids=[rank]), DDP(d, device_ids=[rank])
    opt_g = torch.optim.AdamW(g.parameters(), lr=cfg.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(d.parameters(), lr=cfg.lr * 0.5, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(); cfg.output.mkdir(parents=True, exist_ok=True); history = []
    for epoch in range(1, cfg.epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        totals = torch.zeros(5, device=device)
        for source, target in loader:
            source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fake = g(source); fake_img = torch.cat([fake[:, :1], torch.sigmoid(fake[:, 1:2])], 1)
                real_img = torch.cat([target[:, :1], target[:, 1:2]], 1)
                d_loss = 0.5 * ((d(real_img) - 1).square().mean() + d(fake_img.detach()).square().mean())
            opt_d.zero_grad(set_to_none=True); scaler.scale(d_loss).backward(); scaler.step(opt_d); scaler.update()
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fake = g(source); depth, mask, grad = losses(fake, target)
                adv = (d(torch.cat([fake[:, :1], torch.sigmoid(fake[:, 1:2])], 1)) - 1).square().mean()
                g_loss = 12 * depth + 3 * mask + 0.2 * grad + 0.05 * adv
            scaler.scale(g_loss).backward(); scaler.unscale_(opt_g); torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); scaler.step(opt_g); scaler.update()
            totals += torch.tensor([g_loss.item(), depth.item(), mask.item(), adv.item(), 1], device=device)
        if distributed: dist.all_reduce(totals)
        if not distributed or dist.get_rank() == 0:
            row = {"epoch": epoch, "loss": float(totals[0] / totals[4]), "depth": float(totals[1] / totals[4]), "mask": float(totals[2] / totals[4]), "adv": float(totals[3] / totals[4])}
            history.append(row); print(json.dumps(row), flush=True); (cfg.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                raw = g.module if distributed else g
                torch.save({"state_dict": raw.state_dict(), "epoch": epoch, "config": vars(cfg)}, cfg.output / f"epoch_{epoch:03d}.pt")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
