#!/usr/bin/env python3
"""Adapt the pretrained RangeLDM real prior to native 64-line M1 range images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class M1Ranges(Dataset):
    def __init__(self, directory: Path) -> None:
        self.files = sorted(directory.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(directory)

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as item:
            depth = torch.from_numpy(item["log_range"].astype(np.float32))[None] * 2 - 1
            mask = torch.from_numpy(item["mask"].astype(np.float32))[None]
        image = torch.cat([depth, mask], 0)
        image = torch.roll(image, int(torch.randint(0, image.shape[-1], ()).item()), dims=-1)
        return F.interpolate(image[None], (64, 256), mode="bilinear", align_corners=False)[0]


def sample(vae, unet, scheduler_path: Path, corpus: Path, output: Path, frames: int, steps: int, seed: int, device):
    scheduler = DDIMScheduler.from_pretrained(scheduler_path)
    scheduler.set_timesteps(steps, device=device)
    angles = np.load(corpus / "ring_angles.npy").astype(np.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    target = output / "synthetic"; target.mkdir(parents=True, exist_ok=True)
    counts = []
    vae.eval(); unet.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(frames):
            latent = torch.randn((1, 4, 8, 32), generator=generator, device=device)
            for t in scheduler.timesteps:
                latent = scheduler.step(unet(latent, t).sample, t, latent).prev_sample
            decoded = vae.decode(latent / vae.config.scaling_factor).sample.float()
            decoded = F.interpolate(decoded, (64, 1024), mode="bilinear", align_corners=False)[0]
            depth = torch.clamp((decoded[0] + 1) / 2, 0, 1).cpu().numpy()
            mask = torch.sigmoid(decoded[1]).cpu().numpy() >= 0.5
            ranges = (np.power(2.0, depth * 8.0) - 1.0) * mask
            az = ((np.arange(1024) + .5) / 1024 * 2 * np.pi - np.pi)[None]
            el = angles[:, None]
            xyz = np.stack((ranges * np.cos(el) * np.cos(az), ranges * np.cos(el) * np.sin(az), np.broadcast_to(ranges * np.sin(el), ranges.shape)), -1)
            points = xyz[mask].astype(np.float32); points.tofile(target / f"{i+1:04d}.bin"); counts.append(len(points))
    return {"frames": frames, "median_points": int(np.median(counts)), "total_points": int(sum(counts))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True); p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--unet", type=Path, required=True); p.add_argument("--scheduler", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--sample-frames", type=int, default=50); p.add_argument("--sample-steps", type=int, default=50); p.add_argument("--seed", type=int, default=20260729)
    c = p.parse_args(); distributed = "LOCAL_RANK" in os.environ
    if distributed: dist.init_process_group("nccl"); local_rank = int(os.environ["LOCAL_RANK"])
    else: local_rank = 0
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank); rank = dist.get_rank() if distributed else 0; torch.manual_seed(c.seed + rank)
    dataset = M1Ranges(c.corpus / "train"); sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=c.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True, num_workers=c.workers, pin_memory=True, persistent_workers=c.workers>0)
    vae = AutoencoderKL.from_pretrained(c.vae).to(device).eval(); vae.requires_grad_(False)
    unet = UNet2DModel.from_pretrained(c.unet).to(device)
    wrapped = DistributedDataParallel(unet, device_ids=[local_rank]) if distributed else unet
    noise = DDPMScheduler.from_pretrained(c.scheduler); opt = torch.optim.AdamW(wrapped.parameters(), lr=c.lr, weight_decay=1e-4); scaler = torch.cuda.amp.GradScaler(); c.output.mkdir(parents=True, exist_ok=True); history=[]
    for epoch in range(1, c.epochs+1):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train(); accum=torch.zeros(2, device=device)
        for image in loader:
            image=image.to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                latent=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
            eps=torch.randn_like(latent); t=torch.randint(0,noise.config.num_train_timesteps,(len(image),),device=device,dtype=torch.long); noisy=noise.add_noise(latent,eps,t)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16): loss=F.mse_loss(wrapped(noisy,t).sample.float(),eps.float())
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1); scaler.step(opt); scaler.update(); accum += torch.tensor([loss.item(),1],device=device)
        if distributed: dist.all_reduce(accum)
        if rank==0:
            row={"epoch":epoch,"loss":float(accum[0]/accum[1])}; history.append(row); print(json.dumps(row),flush=True); (c.output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            if epoch % c.save_every ==0 or epoch==c.epochs: (wrapped.module if distributed else wrapped).save_pretrained(c.output/f"epoch_{epoch:03d}",safe_serialization=True)
    if distributed: dist.barrier()
    if rank==0:
        raw=wrapped.module if distributed else wrapped; report=sample(vae,raw,c.scheduler,c.corpus,c.output,c.sample_frames,c.sample_steps,c.seed+99,device); report.update({"method":"M2-pretrained RangeLDM prior fine-tuned on native M1", "epochs":c.epochs}); (c.output/"generation_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report),flush=True)
    if distributed: dist.destroy_process_group()

if __name__ == "__main__": main()
