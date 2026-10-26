# ECCV Submission Reproduction Package

This self-contained package includes data-generation code, detector training
code, six trained checkpoints, and inference, fusion, and packaging scripts.
All project paths are relative. It deliberately excludes derived datasets such
as d50k, rendered batches, synthetic submission frames, and cached predictions.
After obtaining the source data described in
[`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md), the submission can
be reproduced from scratch.

## Environment

- Conda environment: `eccv-v2x-submission` (Python 3.10, PyTorch 2.1.2 CUDA
  11.8, and spconv CUDA 11.8); see `conda_environment.yml`.
- RangeLDM training also requires the `eccv-rangeldm` environment. Its code
  is in `third_party/RangeLDM`.
- The supplied inference workflow uses GPU device 0 and runs the six models
  sequentially.

## Quick reproduction

```bash
cd v2x_submission
bash scripts/run_submission_reproduction.sh
```

Before running, place the 50 base realism frames under
`outputs/current_submission_v2_synthetic/synthetic/`, as generated in steps 2
and 7 of the data-pipeline guide. The script applies the S12-x092 transform,
runs six model inferences, applies v17 fusion, packages the submission, and
checks the expected detection counts and first-frame hash.

The script accepts neutral declaration metadata through environment variables:
`SUBMISSION_AUTHOR`, `SUBMISSION_DATE`, and `SUBMISSION_TAG`.

## End-to-end pipeline

See [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md).

```
source data -> G0S corpus -> RangeLDM rendering batch B -> d50k -> PedAlign v3
            -> six DSVT models -> inference / fusion / realism / package
```

## Layout

```
v2x_submission/
├── PROJECT_OVERVIEW.md / conda_environment.yml
├── docs/REPRODUCTION_GUIDE.md        # complete pipeline guide
├── configs/                         # six model configurations
├── src/                             # data, generation, inference, and packaging tools
├── scripts/                          # preparation, RangeLDM, and submission scripts
├── weights/                         # six checkpoints with portable, short paths
├── competition/starting_kit/        # official test frames and scoring program
└── third_party/                     # OpenPCDet and RangeLDM dependencies
```

## Verification targets

- Detection output: 4,177 boxes: 1,157 `vehicle`, 2,320 `pedestrian`, and 700
  `truck`.
- Realism output: S12-x092 frame `0001.bin` SHA-256 starts with `3446f74d`.
- Optionally place the reference `predictions_final_v17.json` in
  `outputs/final_fusion/` to enable per-box comparison.
