# Reproduction from source data to final submission

This repository does not include pre-generated derivative data (d50k, rendered
batches, synthetic submission frames, or cached predictions). The procedure
below regenerates the training data, trains six detection models, and produces
the final submission with `scripts/run_submission_reproduction.sh`.

## 0. Dependency overview

```
Official UrbanTwin digital-twin data -> 10k base corpus + sequence split (G0S)
                                            |
Official public V2X-Real M1 data -----------+-> RangeLDM S2R render, batch B (2k)
                                                -> d50k (50,000 points/frame, 2k)
                                                -> PedAlign v3 training recipe
                                                -> train six DSVT models
                                                -> inference, v17 fusion, S12-x092, package
```

## 1. Environment and source data

- Create the `eccv-v2x-submission` environment from `conda_environment.yml` (Python
  3.10, PyTorch 2.1.2 CUDA 11.8, and spconv CUDA 11.8).
- RangeLDM training uses `eccv-rangeldm`; see `third_party/RangeLDM/README.md`.
- Download source data through the official channels; it is not distributed here.
  - Place UrbanTwin archives in `data/urbantwin_parts/`.
  - Place public V2X-Real training LiDAR in `data/v2xreal_public/`.
  - The competition starter kit, including test frames and declaration template,
    is in `competition/starting_kit/`.

## 2. Base corpus and sequence split (S1)

```bash
bash scripts/prepare_source_data.sh
python src/analyze_dataset.py ...
python src/build_sequence_splits_v2.py ...
```

This produces the 10k-frame base corpus, `data/openpcdet_g0s_seq250/`, and the
training and validation manifests. The split uses consecutive 250-frame
sequences to prevent leakage.

## 3. RangeLDM rendering (S2, batch B)

```bash
python src/build_rangeldm_s2r_dataset.py ...
bash scripts/train_rangeldm_vae.sh
bash scripts/train_rangeldm_latent_diffusion.sh
python src/infer_rangeldm_sdedit.py --batch B ...
```

The final command produces two 2k-frame batches; the pipeline uses batch B.
Labels are retained from the corresponding source frame. Output:
`data/rangeldm_s2r_2k_b/`.

## 4. d50k density-stable data (S3)

Resample every rendered batch-B frame to exactly 50,000 float32 XYZ points:

```bash
python src/build_detector_rangeldm_dataset.py --points-per-frame 50000 ...
```

The equivalent NumPy operation is
`rng.choice(points, 50000, replace=len(points) < 50000)`. Output:
`data/openpcdet_g0s_seq250_rangeldm2k_b_d50k/`.

## 5. PedAlign v3 augmentation (S4)

The packaged `v2x_ped_align` augmentor in
`third_party/OpenPCDet/pcdet/datasets/augmentor/data_augmentor.py` aligns
pedestrian dimensions and applies structured sparsification. Enable it through
the `configs/dsvt_pedalign*.yaml`
configurations. Its targets are `[0.68, 0.69, 1.74]` m with sparse probability
0.6, one to two angular sectors, and 20–45% random points.

## 6. Train six DSVT models

Use `third_party/OpenPCDet/tools/train.py`. The supplied `weights/` directory
contains the six matching checkpoints, so training can be skipped for quick
reproduction.

| Model | Training data / configuration | Epochs | Reference resources |
|---|---|---:|---|
| d14 | rendered batch B | 40 | 8 GPUs, batch 64 |
| d50 | d50k | 40 | 8 GPUs, batch 64 |
| b4 | d50k + PedAlign v3 | 60 | 6 GPUs, batch 48 |
| b7 | b4 + corner loss 0.3 | 60 | 4 GPUs, batch 32 |
| b10 | b4 + vehicle corner loss 0.15 | 60 | 4 GPUs, batch 32 |
| b11 | b4 + center weights 2/2/1.5 | 60 | 4 GPUs, batch 32 |

```bash
conda run -n eccv-v2x-submission python third_party/OpenPCDet/tools/train.py \
  --launcher pytorch --cfg_file configs/<configuration>.yaml \
  --batch_size <batch-size> --epochs <epochs> --fix_random_seed \
  --extra_tag <tag>
```

Only synthetic labels are used for training. Real frames are limited to
validation and model selection, and every frame listed in `forbidden_frames.txt`
must be excluded.

## 7. Inference, fusion, realism, and package

```bash
bash scripts/run_submission_reproduction.sh
```

Prerequisite: place 50 realism base frames in
`outputs/current_submission_v2_synthetic/synthetic/`. The runner then:

- applies S12-x092 (`x=0.92`, `y=1.00`, `z=0.85`) to create 50 submission
  frames of 9,000 points;
- runs `infer-test` on the official 50 test frames for all six models;
- applies the preserved v17 fusion: four-model vehicle NMS at 0.5, d14+d50
  truck NMS at 0.2, b4 pedestrians, center blending at 0.2/0.2/0.25, and final
  NMS at `0.2 @ 0.03`;
- packages and verifies 4,177 boxes with class counts 1,157 / 2,320 / 700.

## 8. Verification targets

- Total detections: 4,177 (`vehicle` 1,157; `pedestrian` 2,320; `truck` 700).
- S12-x092 `0001.bin` SHA-256 begins with `3446f74d`.
- If the prior `predictions_final_v17.json` is placed under
  `outputs/final_fusion/`, the runner compares every box with tolerances
  `1e-3` for coordinates and `1e-4` for scores.
