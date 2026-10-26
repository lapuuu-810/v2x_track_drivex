#!/usr/bin/env python3
"""Train a conditional range-image renderer from static maps to real returns."""
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


class PairDataset(Dataset):
    def __init__(self, root: Path, subset: str) -> None:
        self.files = sorted((root / subset).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"no {subset} pairs in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[i]) as d:
            sl = torch.from_numpy(d["static_log"].copy()).float()
            sm = torch.from_numpy(d["static_mask"].copy()).float()
            tl = torch.from_numpy(d["target_log"].copy()).float()
            tm = torch.from_numpy(d["target_mask"].copy()).float()
        # A fixed coordinate channel lets the renderer model known ring/azimuth
        # differences without leaking a target frame.
        h, w = sl.shape
        yy = torch.linspace(-1, 1, h)[:, None].expand(h, w)
        xx = torch.linspace(-1, 1, w)[None, :].expand(h, w)
        x = torch.stack((sl * 2 - 1, sm, yy, xx))
        y = torch.stack((tl, tm))
        return x, y


def block(inp: int, out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(max(1, min(16, out // 4)), out),
        nn.SiLU(inplace=True),
        nn.Conv2d(out, out, 3, padding=1, bias=False),
        nn.GroupNorm(max(1, min(16, out // 4)), out),
        nn.SiLU(inplace=True),
    )


class RendererUNet(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.e1 = block(4, width)
        self.e2 = block(width, width * 2, 2)
        self.e3 = block(width * 2, width * 4, 2)
        self.e4 = block(width * 4, width * 8, 2)
        self.mid = block(width * 8, width * 8, 2)
        self.d4 = block(width * 8 + width * 8, width * 8)
        self.d3 = block(width * 8 + width * 4, width * 4)
        self.d2 = block(width * 4 + width * 2, width * 2)
        self.d1 = block(width * 2 + width, width)
        self.out = nn.Conv2d(width, 2, 1)

    @staticmethod
    def up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3); m = self.mid(e4)
        d4 = self.d4(torch.cat((self.up(m, e4), e4), 1))
        d3 = self.d3(torch.cat((self.up(d4, e3), e3), 1))
        d2 = self.d2(torch.cat((self.up(d3, e2), e2), 1))
        d1 = self.d1(torch.cat((self.up(d2, e1), e1), 1))
        return self.out(d1)


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    depth = torch.sigmoid(pred[:, 0])
    mask = target[:, 1]
    # Range is meaningful only where the real sensor returned a point.  A weak
    # background term keeps hallucinated depths bounded when masks are wrong.
    depth_l1 = (torch.abs(depth - target[:, 0]) * (0.15 + 0.85 * mask)).mean()
    mask_bce = F.binary_cross_entropy_with_logits(pred[:, 1], mask)
    # Reward local range gradients to avoid the overly smooth voxel-map look.
    gx = torch.abs(depth[:, :, 1:] - depth[:, :, :-1] - (target[:, 0, :, 1:] - target[:, 0, :, :-1])).mean()
    gy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :] - (target[:, 0, 1:, :] - target[:, 0, :-1, :])).mean()
    total = depth_l1 + 0.5 * mask_bce + 0.15 * (gx + gy)
    return total, {"depth": depth_l1.detach(), "mask": mask_bce.detach(), "grad": (gx + gy).detach()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260805)
    a = p.parse_args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    else:
        rank = local = 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    train = PairDataset(a.data, "train")
    valid = PairDataset(a.data, "validation")
    sampler = DistributedSampler(train, shuffle=True) if distributed else None
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0, drop_last=True)
    model: nn.Module = RendererUNet(a.width).to(device)
    if distributed:
        model = DDP(model, device_ids=[local], broadcast_buffers=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    a.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, a.epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        model.train(); running = torch.zeros(5, device=device)
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                total, terms = loss_fn(model(x), y)
            scaler.scale(total).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            running += torch.stack((total.detach(), terms["depth"], terms["mask"], terms["grad"], torch.ones((), device=device)))
        schedule.step()
        if distributed: dist.all_reduce(running)
        if rank == 0:
            row = {"epoch": epoch, "loss": float(running[0] / running[4]), "depth": float(running[1] / running[4]), "mask": float(running[2] / running[4]), "grad": float(running[3] / running[4]), "lr": opt.param_groups[0]["lr"]}
            history.append(row); print(json.dumps(row), flush=True)
            (a.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch % a.save_every == 0 or epoch == a.epochs:
                raw = model.module if distributed else model
                torch.save({"model": raw.state_dict(), "epoch": epoch, "width": a.width, "config": vars(a)}, a.output / f"epoch_{epoch:03d}.pt")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
