# Multi-task depth estimation and semantic segmentation with DINOv2 + DPT

**ML4CV project (Machine Learning for Computer Vision), University of Bologna**
**Author:** Christian Joseph Fiore

## Purpose

This project builds one network that predicts **metric depth and semantic segmentation in a single forward pass**. It combines a self-supervised **DINOv2 ViT-S/14** backbone, partially fine-tuned, with a **DPT** decoder shared by both tasks, and compares it with the multi-task model **SwinMTL** on **Cityscapes** and **NYUv2**, under SwinMTL's training data and evaluation protocols. It is also compared with DINOv2 and DPT.

The method, the results and all ablations are described in the report (`report.pdf`).

## Setup

1. **Environment.** Tested with Python 3.14, PyTorch 2.11 (CUDA 12.8) on one NVIDIA RTX A4000 (16 GB).
   ```bash
   pip install -r requirements.txt
   ```
   PyTorch comes from the CUDA 12.8 wheel index; change the index URL in the file for another CUDA version.

2. **Third-party code**, cloned into the project folder:
   ```bash
   git clone https://github.com/facebookresearch/dinov2.git
   git clone https://github.com/isl-org/DPT.git
   git clone https://github.com/PardisTaghavi/SwinMTL
   ```

3. **Data.** `data/` is built by `experiments/01_data.ipynb` (downloads, Cityscapes depth ground truth with CREStereo, NYUv2 and the zero-shot datasets; ≈ 4 h on a GPU). Cityscapes may not be redistributed: register at cityscapes-dataset.com and set `CITYSCAPES_USERNAME` / `CITYSCAPES_PASSWORD` before running the notebook. If you already have a copy of `data/`, skip this step.

4. **Checkpoints.** Needed only to reproduce the results without training; place them in `ckpt/`.

## How to run

**Reproduce the results from the checkpoints** (evaluation only, no training, no internet access needed). Open the two notebooks in Jupyter and run all cells, or run them headless from the project folder:

```bash
papermill benchmarks.ipynb benchmarks.ipynb -k <kernel>    # ≈ 1.5–2 h (≈ 15 min with -p TTA False -p ZERO_SHOT False)
papermill ablations.ipynb  ablations.ipynb  -k <kernel>    # ≈ 1–1.5 h (-p RES False skips the resolution study)
```

They are independent and write their outputs to `results/reproduce/`. Evaluations are cached, so a second run only rebuilds the tables (`FORCE = True` recomputes them). The last section of each notebook checks the reproduced numbers against the original ones in `results/`.

**Retrain from scratch.** Run the notebooks in `experiments/` in order, from `01` (or from `02` if `data/` is already there). Each training run resumes after an interruption and is skipped if its checkpoint already exists. One run takes ≈ 8 h on Cityscapes and ≈ 3–3.5 h on NYUv2 on the RTX A4000.

## Licence

This project is released under the **GNU General Public License v3.0** (see `LICENSE`). It builds on three open-source projects, which are cloned separately:

| Project | Licence | Use |
|---|---|---|
| [SwinMTL](https://github.com/PardisTaghavi/SwinMTL) | GPL-3.0 | Parts of its code are ported in `mtl_train.py` and `mtl_common.py` (critic, SiLog loss, patch mixing, photometric augmentation, SwinMTL-compatible metrics); this is why the project is GPL-3.0. |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | The ViT-S/14 backbone and its pretrained weights. |
| [DPT](https://github.com/isl-org/DPT) | MIT | The decoder design, reimplemented. |

The Cityscapes depth ground truth is generated with [CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch), cloned by `experiments/01_data.ipynb`. The datasets (Cityscapes, NYUv2, KITTI, SUN RGB-D, ADE20K, Pascal VOC, Pascal Context) are not covered by this licence and are not redistributed; each has its own terms of use.
