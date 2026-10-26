#!/usr/bin/env python3
"""Sample a saved conditional M1 range-diffusion checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import UNet2DModel

from train_m1_conditional_diffusion import sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-data", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--threshold", type=float, default=.50)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = torch.device(a.device)
    model = UNet2DModel.from_pretrained(a.checkpoint).to(device).eval()
    sample(model, a.seed_data, a.corpus, a.output, a.steps, a.threshold, a.seed, device)


if __name__ == "__main__":
    main()
