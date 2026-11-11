# Multi-task depth estimation and semantic segmentation with DINOv2 + DPT

**ML4CV project (Machine Learning for Computer Vision), University of Bologna**
**Author:** Christian Joseph Fiore

---

## Purpose

This project builds **one network that predicts metric depth and semantic segmentation in a single forward pass**, and checks whether a self-supervised foundation backbone can beat a state-of-the-art multi-task model trained with a task-specific recipe.

- **Model:** a **DINOv2 ViT-S/14** backbone (22 M parameters, self-supervised pre-training) with a shared **DPT** decoder and two light heads, one for depth and one for segmentation.
- **Reference model to beat:** **SwinMTL** (Swin-V2-B, 87.4 M parameters) on Cityscapes and NYUv2, using SwinMTL's training strategy and evaluation protocols.
- **Secondary references:**
  - **DINOv2 ViT-S/14**: Cityscapes segmentation, and NYUv2 depth with a DPT head;
  - **DPT-Hybrid**: NYUv2 depth.
- **Constraints:**
  - NYUv2 is trained on the **795 official labelled images only**. DINOv2 trains its NYU depth head on about 24k raw frames; DPT-Hybrid is depth-pretrained on about 1.4 M images (MIX 6) and then fine-tuned on NYUv2.
  - The headline numbers are **single pass**. Test-time augmentation is reported as supplementary.
  - The Cityscapes training strategy stays SwinMTL's: 512×1024 base, 448×896 crops.
- **Method:** every design choice was decided by an **ablation**. Each ablation changes one factor against a fixed baseline and is judged with a decision rule fixed before the results (below). The final model is trained with **3 seeds** and reported as mean ± std.

---

## Model

| Part | Choice |
|---|---|
| Backbone | DINOv2 ViT-S/14; features from layers {3, 6, 9, 12}; last **8 blocks fine-tuned** with layer-wise LR decay 0.9 |
| Decoder | DPT: reassemble + 4 RefineNet fusion blocks (256 ch), shared by both tasks |
| Depth head | DPT monodepth head, linear output d = d_max · σ(z) |
| Segmentation head | DPT segmentation head + auxiliary head (weight 0.4) |
| Loss | SiLog (λ = 0.5) + cross-entropy + 0.4 · auxiliary cross-entropy |
| Optimiser | AdamW, head LR 1e-4, backbone LR ×0.1 with layer decay; warm-up + poly schedule; batch 8, bf16 |
| Augmentation | photometric, random rescale s ∈ [1, 1.5] with depth ÷ s, random crop, flip, SwinMTL's patch mixing |
| Iterations | Cityscapes 40k (crop 448×896), NYUv2 20k (crop 476×630) |
| Our addition | **Frame-consistent position embeddings** (Cityscapes): during training a crop receives the window of the position embeddings of the full frame, so training and test use the same embedding for the same image position. No parameters, no extra compute. |
| Size | 37.8 M parameters (30.0 M trainable), 475.7 GMACs at 512×1024, 22.6 FPS (RTX A4000, bf16) |

---

## Data

| | Cityscapes | NYUv2 |
|---|---|---|
| Train / test | 2975 / 500 (val used as test, as in SwinMTL) | 795 / 654 (official labelled split) |
| Depth GT | Regenerated from the stereo pairs with **CREStereo** (SwinMTL's method; their maps are not released), clipped to 80 m | Kinect, in-painted |
| Segmentation GT | 19 classes | 40 classes |

Zero-shot evaluation only (no training):
- **Depth:** KITTI Eigen test, SUN RGB-D test.
- **Segmentation:** ADE20K, VOC 2012, Pascal Context, on mapped class subsets.

---

## Results (final model, 3 seeds, single pass)

### Cityscapes

| | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ1 ↑ | RMSE full res ↓ | mIoU ↑ | mIoU multi-scale ↑ |
|---|---|---|---|---|---|---|---|---|
| SwinMTL | **0.089** | 1.051 | 5.481 | **0.139** | **0.921** | 6.352 | 76.41 | — |
| DINOv2 ViT-S/14 | — | — | — | — | — | — | 66.6 (linear) | 77.1 |
| **Ours** | 0.108 ± 0.002 | **0.823 ± 0.003** | **5.024 ± 0.019** | 0.146 ± 0.001 | 0.914 | **5.059 ± 0.019** | **77.74 ± 0.19** | **79.79 ± 0.13** |
| *Ours + sliding window (TTA)* | *0.083* | *0.676* | *4.650* | *0.131* | *0.927* | *4.69* | *78.98* | — |

### NYUv2

| | RMSE ↓ | AbsRel ↓ | δ1 ↑ | mIoU ↑ |
|---|---|---|---|---|
| SwinMTL | 0.518 | — | — | 58.14 |
| DINOv2 ViT-S/14 + DPT (≈ 24k frames) | **0.356** | — | — | — |
| DPT-Hybrid (pretrained on ≈ 1.4 M images, fine-tuned on NYUv2) | 0.357 | **0.110** | **0.904** | — |
| **Ours (795 images)** | **0.420 ± 0.001** | 0.120 ± 0.001 | 0.865 ± 0.001 | 58.10 ± 0.20 (standard) / **58.43 ± 0.22** (SwinMTL's per-image metric) |
| *Ours + sliding window (TTA)* | *0.410* | *0.116* | *0.874* | *59.76* |

### Against the targets

| Target | Result |
|---|---|
| Cityscapes RMSE, SqRel, full-resolution RMSE vs SwinMTL | ✅ beaten on all 3 seeds |
| Cityscapes mIoU vs SwinMTL (76.41), standard metric | ✅ 77.74, all seeds |
| Cityscapes mIoU vs SwinMTL (76.41), SwinMTL's per-image metric | ❌ 65.23, all seeds below (their number is not reproducible from their code: label bug) |
| Cityscapes mIoU vs DINOv2's frozen-feature evaluation (66.6 linear / 77.1 multi-scale) | ✅ 77.74 / 79.79 (a fine-tuned model vs a frozen probe) |
| NYUv2 RMSE vs SwinMTL (0.518) | ✅ 0.420 (−19 %), all seeds |
| NYUv2 mIoU vs SwinMTL (58.14) | tie with the standard metric (58.10, 2/3 seeds above); ✅ with SwinMTL's per-image metric (58.43, all seeds) |
| Cityscapes AbsRel / RMSElog / δ1 vs SwinMTL | ❌ global scale bias (see Findings) |
| NYUv2 depth vs DINOv2 / DPT (0.356 / 0.357) | ❌ training data: 795 images vs ≈ 24k (DINOv2) / ≈ 1.4 M + NYUv2 (DPT) (see ablation D1) |

SwinMTL's mIoU is compared with both metrics on both datasets, including the per-image comparison we lose (Cityscapes). The standard metric is the comparable one.

All with **0.43× SwinMTL's parameters**. Compute: 475.7 GMACs at 512×1024. SwinMTL reports 65 G FLOPs without stating the input size; a Swin-B backbone alone is ≈ 160 G at 512×1024, so the two FLOPs numbers are not directly comparable.

**Zero-shot (reference only):**
- KITTI (Cityscapes model): AbsRel 0.151 raw, 0.112 median-scaled.
- SUN RGB-D (NYUv2 model): RMSE 0.608 raw, 0.515 median-scaled.

---

## Ablations

### Decision rule
- σ is the NYUv2 seed standard deviation, with floors: 0.003 m RMSE, 0.2 mIoU, 1 % relative AbsRel.
- z = Δ / σ, with z > 0 meaning better.
- A change is **adopted** if one task reaches z ≥ +2.5 and no task falls to z ≤ −2.5.
- Ablations use one seed (42), except where marked.

### Summary

| ID | Change | Data | Result | Decision |
|---|---|---|---|---|
| A/B | frozen backbone → last 4 blocks fine-tuned | CS + NYU | mIoU +2.3 (CS), +3.4 (NYU); depth better on both | fine-tune |
| E1 | 4 → 8 fine-tuned blocks (3 + 3 seeds) | NYU | RMSE 0.430 → **0.420**, mIoU 57.39 → **58.10** (z 3.3 / 3.5) | **8 blocks** |
| E2 | WGAN-GP critic (SwinMTL) on → off | NYU | no significant change (z +1.6 / +0.6); training **1.8× faster** | critic off |
| AB1 | standard → **frame-consistent position embeddings** | CS | AbsRel 0.122 → **0.108**, RMSE 5.42 → 5.18; mIoU unchanged | on (Cityscapes) |
| FPE-NYU | frame-consistent PE on → off | NYU | RMSE 0.427 → 0.420, mIoU +0.5 | off on NYU (the crop ≈ the full frame there) |
| AUG-A | no rescale + smaller crop (SwinMTL geometry) | NYU | clearly worse (z −5.3 / −9.0) | keep rescale [1, 1.5] |
| AUG-B | rescale [1, 1.5] → [0.5, 2] | NYU | mIoU +0.39, RMSE −0.004 (below threshold) | unchanged |
| AUG-C | rotation ±2.5° | NYU | no effect | off |
| AUG-D | fix for the photometric RNG shared by data-loader workers (a bug) | NYU | neutral | bug fix (Cityscapes) |
| LOG | linear → log-space depth output (SwinMTL) | CS + NYU | no gain; slightly worse on CS | linear |
| PM | patch mixing on → off (3 + 3 seeds) | NYU | RMSE 0.420 → **0.404** (z +5.4); standard mIoU unchanged; per-image mIoU 58.43 → 57.62 | kept on (keeps the per-image mIoU win over SwinMTL); reported as a trade-off |
| ST-D | multi-task → depth only | CS + NYU | CS AbsRel 0.106 → 0.100; NYU RMSE 0.420 → 0.383 | stays multi-task (cost of multi-task learning) |
| ST-S | multi-task → segmentation only | CS + NYU | no significant change (−0.2 / −0.1 mIoU) | stays multi-task |
| D1 | DINOv2's setup (frozen ViT-S + DPT, depth only) on our 795 images | NYU | RMSE 0.420 (DINOv2 reports 0.356 with ≈ 24k frames) | the gap is training data |
| D4 | DINOv2's 256-bin depth head | NYU | depth unchanged, mIoU −0.8 | side ablation only |
| RES | inference resolution s ∈ [0.5, 1.5] (evaluation only) | CS + NYU | the metric depth scale shifts with input size; segmentation is robust | protocol stays at s = 1 |

The full numeric table is in `results/compare/all_ablations.csv`.

### Key findings
1. **Partial fine-tuning of DINOv2 matters.** Results improve monotonically from frozen to 4 to 8 blocks, on both tasks and both datasets, with no over-fitting on 795 images.
2. **SwinMTL's WGAN-GP critic and log-space depth output do not transfer** to a DINOv2 backbone. The critic only costs training time.
3. **The Cityscapes AbsRel gap is a global scale bias, not worse structure.**
   - Predictions are ≈ 5 % too far. After removing only the global scale, AbsRel is ≈ 0.092, at SwinMTL's level.
   - The cause is training on 448×896 crops and testing on the full 512×1024 frame with a global-attention ViT and absolute position embeddings.
   - Evidence: a sliding window at the training crop size removes the gap (0.083); frame-consistent PE removes part of it (AB1); NYUv2, where the crop ≈ the full frame, has almost no bias.
4. **The NYUv2 depth gap to DINOv2 / DPT is training data.** DINOv2's own setup on our 795 images reaches the same 0.420 (D1), and the head is not the cause (D4).
5. **Multi-task learning costs depth and gives no segmentation gain** with this backbone (ST-D / ST-S). This is the opposite of SwinMTL's ablation. The benefit of the multi-task model is efficiency: one backbone pass for two tasks.
6. **SwinMTL's patch mixing is a depth/segmentation trade-off** on NYUv2: removing it improves depth by 3.8 % but lowers the per-image mIoU.
7. **The metric depth scale depends on the input resolution** for this ViT; segmentation does not.
8. **SwinMTL's reported Cityscapes mIoU (76.41) is not comparable.** Their loader remaps labels in place incorrectly and their code uses a per-image metric. We report the standard dataset-level mIoU.

### Limitations
- Ablations use a single seed, except E1 and PM. The Cityscapes σ is transferred from NYUv2.
- The Cityscapes depth GT is regenerated with CREStereo and cannot be identical to SwinMTL's maps.
- The Cityscapes val set is also the test set, as in SwinMTL. No tuning was done on it: the last iterate is always used.
- Patch mixing was ablated on NYUv2 only.
- Runs are not bit-exact reproducible because some CUDA kernels are non-deterministic.

---

## Repository structure

```
benchmarks.ipynb      ← reproduce all benchmark tables of the final model from the checkpoints
ablations.ipynb       ← reproduce all ablation studies from the checkpoints
repro_common.py       ← evaluation helpers of the two notebooks above
mtl_common.py         ← shared code: configuration, datasets, model, metrics, evaluation
mtl_train.py          ← training code: critic, losses, patch mixing, optimizer, schedule, train_run
mtl_final.py          ← final-model helpers: seed noise σ, decision rule, train_eval (notebooks 09–13)
PAPER_SUMMARY.md      ← detailed summary of every experiment (setup, result, finding, cause, decision)
requirements.txt      ← exact package versions used for all runs
LICENSE               ← GPL-3.0
experiments/          ← development notebooks 01–13: data preparation, training of every run, original analyses
ckpt/                 ← model checkpoints (not in the repository, see below)
results/              ← original evaluations and tables of the development notebooks
data/                 ← prepared datasets (≈ 17 GB, not in the repository)
dinov2/  DPT/  SwinMTL/   ← third-party code (cloned, see below)
```

**Development notebooks** (`experiments/`). Each one trained and analysed part of the work and documents in its markdown what was done, why, and every fix or decision, with dates.

| Notebook | Content |
|---|---|
| `01_data.ipynb` | Data download and preparation into `data/`; Cityscapes depth GT with CREStereo; zero-shot datasets |
| `02_model_training.ipynb` | Model, losses, training loop; baseline runs A (frozen) and B (4 blocks) |
| `03_benchmarks.ipynb` | Original benchmark analysis: SwinMTL, DINOv2 and DPT comparisons, zero-shot, efficiency; 'Final model vs the three papers' = final model |
| `04_training_recipe.ipynb` | E1 (8 blocks), E2 (no critic) |
| `05_ablation_frame_pe.ipynb` | AB1 (frame-consistent position embeddings) |
| `06_seed_repeats.ipynb` | 4 vs 8 blocks on 3 seeds each, and the seed noise σ |
| `07_ablation_augmentation.ipynb` | AUG-A to AUG-D |
| `08_inference_resolution.ipynb` | RES (evaluation only) |
| `09_final_model.ipynb` | Final configuration chosen by the decision rule, LOG ablation, final seeds |
| `10_ablations_final.ipynb` | Ablations on the final model (ST-D, ST-S, PM, FPE-NYU, D4) and the original table of all ablations |
| `11_d1_dinov2_setup.ipynb` | D1 (DINOv2's depth setup on our data) |
| `12_final_nyu_nopm.ipynb` | PM on 3 seeds |
| `13_st_nyu_nopm.ipynb` | Single-task runs without patch mixing (supplementary) |

---

## Reproducing the results (from the checkpoints)

The two notebooks at the root **only evaluate**. They load every model from `ckpt/`, re-run all evaluations and rebuild the tables. Nothing is trained, and they do not depend on the development notebooks.

### 1. What must be in the project folder

| Folder | Content | Needed by |
|---|---|---|
| `ckpt/` | the checkpoints (≈ 20 GB): `final/`, `seeds/` for the final model; all groups for the ablations | both |
| `data/` | the prepared datasets (≈ 17 GB): `cityscapes/` (incl. the CREStereo depth maps) and `nyu/`; plus `kitti/`, `sunrgbd/`, `ade20k/`, `voc/`, `pcontext/` for zero-shot. Built by `experiments/01_data.ipynb`. | both (zero-shot sets: benchmarks only) |
| `dinov2/`, `DPT/`, `SwinMTL/` | third-party repositories (`mtl_common.py` checks that they exist; only DINOv2's model code is used) | both |
| `results/` | *optional*: the original evaluations; only used for the final reproduction check | both |

```bash
git clone https://github.com/facebookresearch/dinov2.git
git clone https://github.com/isl-org/DPT.git
git clone https://github.com/PardisTaghavi/SwinMTL
```

No internet access is needed: the models are built without downloading pretrained weights and loaded from `final.pt`.

**With `data/` and `ckpt/` in place, `experiments/01_data.ipynb` is not needed.** That notebook only builds `data/` from the original datasets: downloads, CREStereo depth for Cityscapes, conversion of NYUv2 and the zero-shot sets. Run it only if you do not have `data/`.

> **Dataset licences.** `data/` contains the original datasets, so whoever shares it must respect their terms. Cityscapes in particular may not be redistributed: each user registers and downloads it from cityscapes-dataset.com. Without a copy of `data/`, run `experiments/01_data.ipynb` with your own Cityscapes account (environment variables `CITYSCAPES_USERNAME` / `CITYSCAPES_PASSWORD`); it rebuilds the same `data/`, including the CREStereo depth maps (≈ 4 h on a GPU).

### 2. Environment
- Tested with Python 3.14, PyTorch 2.11 (CUDA 12.8), one NVIDIA RTX A4000 (16 GB).
- Install the packages with `pip install -r requirements.txt`, which pins the exact versions used for all runs. PyTorch comes from the CUDA 12.8 wheel index; change the index URL in the file for another CUDA version.

### 3. Run
Open the notebooks in Jupyter and run all cells, or run them headless from the project folder:
```bash
papermill benchmarks.ipynb benchmarks.ipynb -k <kernel>    # ≈ 1.5–2 h (≈ 15 min with -p TTA False -p ZERO_SHOT False)
papermill ablations.ipynb  ablations.ipynb  -k <kernel>    # ≈ 1–1.5 h (RES study included; -p RES False skips it)
```
The two are independent and can be run in any order.

| Notebook | Produces (in `results/reproduce/`) |
|---|---|
| `benchmarks.ipynb` | `benchmarks/`: per-seed evaluations; SwinMTL Tab. I/II/III/V; DINOv2 and DPT tables; zero-shot; efficiency; target scorecard |
| `ablations.ipynb` | `ablations/`: per-run evaluations; seed noise σ; table of all ablations with z-scores and decisions; details for AB1, PM, single-task and D1; inference-resolution study |

- Every evaluation is cached as JSON, so a second run only rebuilds the tables. Set `FORCE = True` to recompute.
- The last section of each notebook compares every reproduced number with the original evaluation in `results/` and reports the largest relative difference. Expect about 0 on the same GPU, and about 1e-4 on a different one (floating point).

### Retraining from scratch (optional)
- Run the development notebooks in `experiments/` in order: from 02 if `data/` is already there (02 → 13), otherwise from 01. Run them from the project root, or open them from `experiments/`: their first cell switches to the root.
- Every training run resumes from its last checkpoint after an interruption and is skipped if `final.pt` already exists.
- Training time on the RTX A4000: Cityscapes ≈ 8 h per run, NYUv2 ≈ 3–3.5 h per run, without the critic.

---

## References

- P. Taghavi, R. Langari, G. Pandey, *SwinMTL: A Shared Architecture for Simultaneous Depth Estimation and Semantic Segmentation from Monocular Camera Images*, arXiv:2403.10662, 2024.
- M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- R. Ranftl, A. Bochkovskiy, V. Koltun, *Vision Transformers for Dense Prediction* (DPT), ICCV 2021.
- J. Li et al., *Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation* (CREStereo), CVPR 2022.

---

## Licence

This project is released under the **GNU General Public License v3.0** (see `LICENSE`).

It builds on three open-source projects, which are not included in this repository (they are cloned separately, see "Reproducing the results"):

| Project | Licence | How it is used here |
|---|---|---|
| [SwinMTL](https://github.com/PardisTaghavi/SwinMTL) | GPL-3.0 | Parts of its code are ported in `mtl_train.py` and `mtl_common.py`: the WGAN-GP critic, the SiLog loss, patch mixing, the photometric augmentation and the SwinMTL-compatible depth and segmentation metrics. This is why the project is GPL-3.0. |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | The ViT-S/14 backbone and its pretrained weights, imported from the cloned repository. |
| [DPT](https://github.com/isl-org/DPT) | MIT | The decoder design (reassemble and fusion blocks), reimplemented. |

The Cityscapes depth ground truth is generated with a PyTorch port of CREStereo ([CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)), which `experiments/01_data.ipynb` clones when it is run.

The datasets (Cityscapes, NYUv2, KITTI, SUN RGB-D, ADE20K, Pascal VOC, Pascal Context) are not covered by this licence and are not redistributed here; each has its own terms of use.
