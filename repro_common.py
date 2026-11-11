"""Shared code of the two reproduction notebooks (benchmarks.ipynb, ablations.ipynb).

The reproduction notebooks only *evaluate*: every model is loaded from its checkpoint in ckpt/ and nothing is trained.
"""
from mtl_common import *

# extra variants used by the runs (only the number of fine-tuned blocks differs; the forward pass is the same)
EXTRA_VARIANTS = {"partial_ft8": 8, "final": 8}
REPRO_DIR = ROOT / "results" / "reproduce"


# =========================================================================== loading a run from ckpt/

def setup(**overrides) -> None:
    """configure() for evaluation, plus the extra variants."""
    configure(**overrides)
    for name, blocks in EXTRA_VARIANTS.items():
        CFG["variants"][name] = dict(CFG["variants"]["partial_ft"], n_trainable_blocks=blocks)


def load_run(run: str, dataset: str, **overrides) -> nn.Module:
    """Load the final weights of a run, in eval mode.

    Args:
        run: run folder relative to ckpt/, e.g. "final/cityscapes_base_s42/cityscapes_final". Its last part is
            run_name(dataset, variant), as written by train_run.
        dataset: "cityscapes" or "nyu".
        overrides: the evaluation-relevant settings of the run (frame_pe, depth_param, depth_head).
    """
    setup(**overrides)
    folder = ROOT / "ckpt" / run
    variant = folder.name[len(dataset) + 1:]
    model = build_model(dataset, variant, pretrained=False)
    ckpt = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def original_eval_path(run: str, suffix: str = "") -> Path:
    """Where the development notebooks saved the evaluation of a run: ckpt/<group>/<run> -> results/<group>/eval/<run>.json."""
    parts = Path(run).parts
    return ROOT / "results" / Path(*parts[:-1]) / "eval" / f"{parts[-1]}{suffix}.json"


def cached(path: Path, compute, force: bool = False):
    """Load path if it exists (and not force), else compute, save and return."""
    if path.exists() and not force:
        return load_json(path)
    result = compute()
    save_json(result, path)
    return result


# =========================================================================== paper reference numbers (experiments/03)

PAPER = {
    # SwinMTL Tab. I: single-task depth, Cityscapes "test" (= val), SwinMTL at 512x1024
    "swinmtl_tab1": {
        "GLNet":          dict(res="1024x2048", abs_rel=0.111, sq_rel=None,  rmse=6.437, rmse_log=0.182, d1=0.868, d2=0.961, d3=0.983),
        "Pilzer et al.*": dict(res="512x256",   abs_rel=0.438, sq_rel=5.713, rmse=5.745, rmse_log=0.400, d1=0.711, d2=0.877, d3=0.940),
        "Struct2Depth*":  dict(res="416x128",   abs_rel=0.145, sq_rel=1.737, rmse=7.280, rmse_log=0.205, d1=0.813, d2=0.942, d3=0.976),
        "Li et al.*":     dict(res="416x128",   abs_rel=0.119, sq_rel=1.290, rmse=6.980, rmse_log=0.190, d1=0.846, d2=0.952, d3=0.982),
        "DynamicDepth*":  dict(res="416x128",   abs_rel=0.103, sq_rel=1.000, rmse=5.867, rmse_log=0.157, d1=0.895, d2=0.974, d3=0.991),
        "Lee et al.*":    dict(res="832x256",   abs_rel=0.116, sq_rel=1.213, rmse=6.695, rmse_log=0.186, d1=0.852, d2=0.951, d3=0.982),
        "InstaDM*":       dict(res="832x256",   abs_rel=0.111, sq_rel=1.158, rmse=6.437, rmse_log=0.182, d1=0.868, d2=0.961, d3=0.983),
        "SwinMTL":        dict(res="512x1024",  abs_rel=0.089, sq_rel=1.051, rmse=5.481, rmse_log=0.139, d1=0.921, d2=0.976, d3=0.990),
    },
    # SwinMTL Tab. II: MTL on Cityscapes at full resolution (1024x2048): (RMSE, mIoU)
    "swinmtl_tab2": {
        "Hybridnet": (12.090, 66.61), "MGMNet": (8.300, 55.70), "MTL": (6.797, 70.43), "PAD-Net": (6.777, 70.23),
        "3-ways": (6.528, 75.00), "[44]": (6.649, 74.95), "SwinMTL": (6.352, 76.41),
    },
    # SwinMTL Tab. III: MTL on NYUv2: (RMSE, mIoU)
    "swinmtl_tab3": {
        "Cross-Stitch": (0.6290, 36.34), "MTL": (0.6380, 39.44), "PAD-Net": (0.6270, 36.61), "3-ways": (0.622, 39.47),
        "[44]": (0.6040, 38.93), "MTAN": (0.5906, 40.01), "MTI-Net": (0.5365, 45.97), "ATRC": (0.5363, 46.33),
        "InvPT": (0.5183, 53.56), "TaskPrompter": (0.5152, 55.30), "SwinMTL": (0.5179, 58.14),
    },
    # SwinMTL Tab. V: (FLOPs (G), parameters (M))
    "swinmtl_tab5": {"TaskPrompter": (416, 373), "ATRC w/ ViT-B": (260, 119), "PAD-Net w/ ViT-B": (166, 109), "SwinMTL w/ Swin-B": (65, 87.38)},
    # DINOv2 Tab. 11: depth RMSE with frozen features, ViT-S/14 row: (lin.1, lin.4, DPT)
    "dinov2_tab11_vits14": {"nyu": (0.449, 0.417, 0.356), "kitti": (3.10, 2.86, 2.34), "sun": (0.477, 0.431, 0.409)},
    # DINOv2 Tab. 10: segmentation mIoU with frozen features, ViT-S/14 row: (lin., +ms)
    "dinov2_tab10_vits14": {"ade20k": (44.3, 47.2), "cityscapes": (66.6, 77.1), "voc2012": (81.1, 82.6)},
    # DPT Tab. 2 (NYUv2)
    "dpt_tab2": {
        "DORN": dict(d1=0.828, d2=0.965, d3=0.992, abs_rel=0.115, rmse=0.509, log10=0.051),
        "VNL": dict(d1=0.875, d2=0.976, d3=0.994, abs_rel=0.111, rmse=0.416, log10=0.048),
        "BTS": dict(d1=0.885, d2=0.978, d3=0.994, abs_rel=0.110, rmse=0.392, log10=0.047),
        "DPT-Hybrid": dict(d1=0.904, d2=0.988, d3=0.998, abs_rel=0.110, rmse=0.357, log10=0.045),
    },
    # DPT Tab. 3 (KITTI Eigen)
    "dpt_tab3": {
        "DORN": dict(d1=0.932, d2=0.984, d3=0.994, abs_rel=0.072, rmse=2.626, rmse_log=0.120),
        "VNL": dict(d1=0.938, d2=0.990, d3=0.998, abs_rel=0.072, rmse=3.258, rmse_log=0.117),
        "BTS": dict(d1=0.956, d2=0.993, d3=0.998, abs_rel=0.059, rmse=2.756, rmse_log=0.096),
        "DPT-Hybrid": dict(d1=0.959, d2=0.995, d3=0.999, abs_rel=0.062, rmse=2.573, rmse_log=0.092),
    },
    # DPT Tab. 4 (ADE20K val) and Tab. 5 (Pascal Context val): (pixAcc, mIoU)
    "dpt_tab4": {"DPT-Hybrid": (83.11, 49.02), "DPT-Large": (82.70, 47.63)},
    "dpt_tab5": {"DPT-Hybrid": (84.83, 60.46)},
    # DPT Tab. 9: (parameters (M), latency (ms, RTX 2080, 384x384))
    "dpt_tab9": {"DPT-Hybrid": (123, 38), "DPT-Large": (343, 35)},
}


# =========================================================================== targets of the final model (single place)

# (label, dataset, evaluation key, metric, reference value from PAPER, lower is better)
# Used by benchmarks.ipynb, experiments/03_benchmarks.ipynb and experiments/09_final_model.ipynb.
_S1, _S2, _S3 = PAPER["swinmtl_tab1"]["SwinMTL"], PAPER["swinmtl_tab2"]["SwinMTL"], PAPER["swinmtl_tab3"]["SwinMTL"]
_D10, _D11, _DPT = PAPER["dinov2_tab10_vits14"]["cityscapes"], PAPER["dinov2_tab11_vits14"]["nyu"], PAPER["dpt_tab2"]["DPT-Hybrid"]
FINAL_TARGETS = [
    ("CS AbsRel vs SwinMTL", "cityscapes", "cs_depth_512x1024", "abs_rel", _S1["abs_rel"], True),
    ("CS SqRel vs SwinMTL", "cityscapes", "cs_depth_512x1024", "sq_rel", _S1["sq_rel"], True),
    ("CS RMSE (512) vs SwinMTL", "cityscapes", "cs_depth_512x1024", "rmse", _S1["rmse"], True),
    ("CS RMSElog vs SwinMTL", "cityscapes", "cs_depth_512x1024", "rmse_log", _S1["rmse_log"], True),
    ("CS δ1 vs SwinMTL", "cityscapes", "cs_depth_512x1024", "d1", _S1["d1"], False),
    ("CS RMSE (full) vs SwinMTL", "cityscapes", "cs_depth_full", "rmse", _S2[0], True),
    ("CS mIoU (standard) vs SwinMTL", "cityscapes", "cs_seg", "mIoU", _S2[1], False),
    ("CS mIoU (SwinMTL per-image) vs SwinMTL", "cityscapes", "cs_seg_swinmtl", "mIoU", _S2[1], False),
    ("CS mIoU vs DINOv2 (frozen, lin.)", "cityscapes", "cs_seg", "mIoU", _D10[0], False),
    ("CS mIoU +ms (TTA, both sides) vs DINOv2 (frozen, +ms)", "cityscapes", "cs_seg_ms", "mIoU", _D10[1], False),
    ("NYU RMSE vs SwinMTL", "nyu", "nyu_depth", "rmse", _S3[0], True),
    ("NYU mIoU (standard) vs SwinMTL", "nyu", "nyu_seg", "mIoU", _S3[1], False),
    ("NYU mIoU (SwinMTL per-image) vs SwinMTL", "nyu", "nyu_seg_swinmtl", "mIoU", _S3[1], False),
    ("NYU RMSE vs DINOv2 + DPT", "nyu", "nyu_depth", "rmse", _D11[2], True),
    ("NYU RMSE vs DPT-Hybrid", "nyu", "nyu_depth", "rmse", _DPT["rmse"], True),
    ("NYU AbsRel vs DPT-Hybrid", "nyu", "nyu_depth", "abs_rel", _DPT["abs_rel"], True),
    ("NYU δ1 vs DPT-Hybrid", "nyu", "nyu_depth", "d1", _DPT["d1"], False),
]


# =========================================================================== zero-shot evaluation (experiments/03)

# target class -> source classes
TRANSFER = {
    ("cityscapes", "ade20k"): {
        "road": ["road"], "sidewalk": ["sidewalk"], "building": ["building"], "fence": ["fence"],
        "pole": ["pole"], "traffic light": ["traffic light"], "signboard": ["traffic sign"], "tree": ["vegetation"],
        "sky": ["sky"], "person": ["person", "rider"], "car": ["car"], "truck": ["truck"], "bus": ["bus"],
        "minibike": ["motorcycle"], "bicycle": ["bicycle"],
    },
    ("nyu", "ade20k"): {
        "wall": ["wall"], "floor": ["floor"], "cabinet": ["cabinet"], "bed": ["bed"], "chair": ["chair"],
        "sofa": ["sofa"], "table": ["table"], "door": ["door"], "windowpane": ["window"], "bookcase": ["bookshelf"],
        "painting": ["picture"], "counter": ["counter"], "blind": ["blinds"], "desk": ["desk"], "shelf": ["shelves"],
        "curtain": ["curtain"], "chest of drawers": ["dresser"], "pillow": ["pillow"], "mirror": ["mirror"],
        "rug": ["floor mat"], "ceiling": ["ceiling"], "book": ["books"], "refrigerator": ["refridgerator"],
        "television receiver": ["television"], "towel": ["towel"], "box": ["box"], "person": ["person"],
        "toilet": ["toilet"], "sink": ["sink"], "lamp": ["lamp"], "bathtub": ["bathtub"], "bag": ["bag"],
    },
    ("cityscapes", "voc2012"): {
        "bicycle": ["bicycle"], "bus": ["bus"], "car": ["car"], "motorbike": ["motorcycle"],
        "person": ["person", "rider"], "train": ["train"],
    },
    ("nyu", "voc2012"): {
        "chair": ["chair"], "diningtable": ["table"], "person": ["person"], "sofa": ["sofa"],
        "tvmonitor": ["television"],
    },
    ("cityscapes", "pcontext"): {
        "road": ["road"], "sidewalk": ["sidewalk"], "building": ["building"], "fence": ["fence"],
        "sign": ["traffic sign"], "tree": ["vegetation"], "sky": ["sky"], "person": ["person", "rider"], "car": ["car"],
        "truck": ["truck"], "bus": ["bus"], "motorbike": ["motorcycle"], "bicycle": ["bicycle"], "train": ["train"],
    },
    ("nyu", "pcontext"): {
        "wall": ["wall"], "floor": ["floor"], "cabinet": ["cabinet"], "bed": ["bed"], "chair": ["chair"],
        "sofa": ["sofa"], "table": ["table", "desk"], "door": ["door"], "window": ["window"],
        "shelves": ["shelves", "bookshelf"], "curtain": ["curtain"], "ceiling": ["ceiling"], "book": ["books"],
        "tvmonitor": ["television"], "person": ["person"], "bag": ["bag"], "cloth": ["clothes"], "light": ["lamp"],
    },
}

# target dataset -> (class names, background class)
TARGETS = {
    "ade20k": (ADE150_NAMES, None),
    "voc2012": (VOC_NAMES, "background"),
    "pcontext": (PC59_NAMES, None),
}


def make_transfer(source: str, target: str) -> Tuple[List[str], Tensor, Tensor]:
    """Look-up tables of a zero-shot transfer: evaluated classes, prediction LUT, GT LUT."""
    source_names = SOURCE_NAMES[source]
    target_names, background = TARGETS[target]
    mapping = TRANSFER[(source, target)]

    classes = list(mapping) + ([background] if background else [])

    # unmapped predictions go to background (VOC) or to an extra "none" class
    none_id = classes.index(background) if background else len(classes)

    source_lut = np.full(len(source_names), none_id, np.int64)
    for target_class, source_classes in mapping.items():
        for source_class in source_classes:
            source_lut[source_names.index(source_class)] = classes.index(target_class)

    target_lut = np.full(256, 255, np.int64)
    for target_class in classes:
        target_lut[target_names.index(target_class)] = classes.index(target_class)

    return classes, torch.from_numpy(source_lut), torch.from_numpy(target_lut)


def to_numpy(x) -> np.ndarray:
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def iterate_eval(kind: str, limit: Optional[int]):
    """Iterate over an eval-only dataset (batch_size=None returns the raw samples)."""
    dataset = make_eval_dataset(kind)
    n = len(dataset) if limit is None else min(limit, len(dataset))
    data_loader = DataLoader(Subset(dataset, range(n)), batch_size=None, num_workers=min(4, CFG["num_workers"]))
    return tqdm(data_loader, total=n, desc=f"zero-shot {kind}", leave=False)


@torch.no_grad()
def eval_kitti_zeroshot(model: nn.Module, limit: Optional[int] = None) -> dict:
    """Cityscapes model on KITTI Eigen: KB crop, focal normalisation, Garg crop, cap 80 m."""
    model.eval()
    raw_meter = DepthMeter(1e-3, 80.0)
    median_meter = DepthMeter(1e-3, 80.0, median_scale=True)
    train_focal = CFG["datasets"]["cityscapes"]["fx_base"]

    for img, gt, focal in iterate_eval("kitti", limit):
        img = to_numpy(img)
        gt = torch.as_tensor(to_numpy(gt), device=device)
        H, W = gt.shape

        # KB crop 352x1216
        top = H - 352
        left = (W - 1216) // 2
        crop = np.ascontiguousarray(img[top:top + 352, left:left + 1216])

        # resize so that the focal length matches the training one
        s = train_focal / float(focal)
        x = prep_image(crop, (int(round(352 * s)), int(round(1216 * s))))
        depth = predict(model, x)[0]
        depth = F.interpolate(depth, size=(352, 1216), mode="bilinear", align_corners=False)[0, 0]

        # paste into a full-size canvas
        full_depth = torch.zeros(H, W, device=device)
        full_depth[top:top + 352, left:left + 1216] = depth

        # Garg crop
        garg_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        garg_mask[int(0.40810811 * H):int(0.99189189 * H), int(0.03594771 * W):int(0.96405229 * W)] = True

        raw_meter.update(full_depth, gt, garg_mask)
        median_meter.update(full_depth, gt, garg_mask)

    return dict(kitti_raw=raw_meter.result(), kitti_median=median_meter.result())


@torch.no_grad()
def eval_sun_zeroshot(model: nn.Module, limit: Optional[int] = None) -> dict:
    """NYU model on SUN RGB-D: focal normalisation, cap 10 m."""
    model.eval()
    raw_meter = DepthMeter(1e-3, 10.0)
    median_meter = DepthMeter(1e-3, 10.0, median_scale=True)
    train_focal = CFG["datasets"]["nyu"]["fx_base"]

    for img, gt, fx in iterate_eval("sunrgbd", limit):
        img = to_numpy(img)
        gt = torch.as_tensor(to_numpy(gt), device=device)
        H, W = gt.shape

        s = train_focal / float(fx)
        depth = predict(model, prep_image(img, (int(round(H * s)), int(round(W * s)))))[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]

        raw_meter.update(depth, gt)
        median_meter.update(depth, gt)

    return dict(sun_raw=raw_meter.result(), sun_median=median_meter.result())


@torch.no_grad()
def eval_seg_transfer(model: nn.Module, source: str, target: str, limit: Optional[int] = None) -> dict:
    """Zero-shot segmentation over the mapped classes (source = training dataset of the model)."""
    model.eval()
    classes, source_lut, target_lut = make_transfer(source, target)
    meter = SegMeter(len(classes) + 1)
    short_side = CFG["datasets"][source]["base_size"][0]   # training scale

    for img, gt in iterate_eval(target, limit):
        img = to_numpy(img)
        gt = torch.as_tensor(to_numpy(gt)).long()
        H, W = gt.shape

        s = short_side / min(H, W)
        probs = predict(model, prep_image(img, (int(round(H * s)), int(round(W * s)))))[1]
        pred = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0).cpu()

        meter.update(source_lut[pred], target_lut[gt.clamp(0, 255)])

    results = meter.result(classes=range(len(classes)))
    return dict(mIoU=results["mIoU"], pixAcc=results["pixAcc"], n_classes=len(classes), classes=classes,
                per_class_iou=results["per_class_iou"])


def zero_shot(model: nn.Module, dataset: str) -> dict:
    """All zero-shot evaluations of a model trained on dataset (as experiments/03 zero_shot_run)."""
    results = eval_kitti_zeroshot(model) if dataset == "cityscapes" else eval_sun_zeroshot(model)
    for target in ["ade20k", "voc2012", "pcontext"]:
        results[f"{target}_subset"] = eval_seg_transfer(model, dataset, target)
    return results


# =========================================================================== log bias (experiments/09)

@torch.no_grad()
def log_bias(model: nn.Module, dataset: str) -> float:
    """Mean log ratio prediction / GT with the standard single-pass protocol (> 0 = predicted too far)."""
    dcfg = CFG["datasets"][dataset]
    ds = make_dataset(dataset, dcfg["eval_split"], train=False)
    out = []
    for i in range(len(ds)):
        sample = ds[i]
        gt = sample["depth"].to(device)
        gt = gt[0] if gt.ndim == 3 else gt
        depth = predict(model, sample["image"][None].to(device))[0]
        if dataset == "cityscapes":
            gt = F.interpolate(gt[None, None], size=depth.shape[-2:], mode="nearest")[0, 0]
            pred = depth[0, 0]
            valid = (gt > dcfg["min_depth"]) & (gt < dcfg["max_depth"])
        else:
            H, W = gt.shape
            pred = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            valid = (gt > dcfg["min_depth"]) & (gt < dcfg["max_depth"]) & eigen_crop_mask(H, W)
        pred = pred.clamp(dcfg["min_depth"], dcfg["max_depth"])
        if valid.any():
            out.append((torch.log(pred[valid]) - torch.log(gt[valid])).mean().item())
    return float(np.mean(out))


# =========================================================================== inference resolution (experiments/08)

@torch.no_grad()
def eval_resolution(model: nn.Module, dataset: str, s: float) -> dict:
    """Single pass with the input resized by s; metrics at the standard evaluation resolution."""
    dcfg = CFG["datasets"][dataset]
    ds = make_dataset(dataset, dcfg["eval_split"], train=False)
    depth_meter = DepthMeter(dcfg["min_depth"], dcfg["max_depth"])
    seg_meter = SegMeter(dcfg["eval_classes"])
    biases = []
    model.eval()
    for i in tqdm(range(len(ds)), desc=f"{dataset} s={s}", leave=False):
        sample = ds[i]
        img = sample["image"][None].to(device)
        gt_depth = sample["depth"].to(device)
        gt_depth = gt_depth[0] if gt_depth.ndim == 3 else gt_depth
        gt_seg = sample["seg"].to(device)
        h, w = img.shape[-2:]
        x = img if s == 1.0 else F.interpolate(img, size=(int(round(h * s)), int(round(w * s))), mode="bilinear", align_corners=False)
        depth, probs = predict(model, x)
        if dataset == "cityscapes":
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)[0, 0]      # 512x1024
            gt_eval = F.interpolate(gt_depth[None, None], size=(h, w), mode="nearest")[0, 0]
            mask = None
            valid = (gt_eval > dcfg["min_depth"]) & (gt_eval < dcfg["max_depth"])
            H, W = gt_seg.shape
            pred_seg = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[0][:dcfg["eval_classes"]].argmax(0)
            seg_meter.update(pred_seg, cs_to_eval(gt_seg))
        else:
            H, W = gt_depth.shape
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            gt_eval = gt_depth
            mask = eigen_crop_mask(H, W)
            valid = (gt_eval > dcfg["min_depth"]) & (gt_eval < dcfg["max_depth"]) & mask
            pred_seg = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0)
            seg_meter.update(pred_seg, gt_seg)
        if mask is None:
            depth_meter.update(depth, gt_eval)
        else:
            depth_meter.update(depth, gt_eval, mask)
        pred = depth.clamp(dcfg["min_depth"], dcfg["max_depth"])
        biases.append((torch.log(pred[valid]) - torch.log(gt_eval[valid])).mean().item())
    d = depth_meter.result()
    return {"abs_rel": d["abs_rel"], "rmse": d["rmse"], "d1": d["d1"], "log_bias": float(np.mean(biases)), "mIoU": seg_meter.result()["mIoU"]}


# =========================================================================== decision rule (experiments/09)

Z = 2.5
SIGMA_FLOOR = dict(rmse=0.003, miou=0.2, abs_rel_cv=0.01)


def seed_sigma(arms: Dict[str, List[dict]]) -> dict:
    """Pooled seed std over the arms (lists of NYU evaluation dicts), with the floors.

    Same computation as experiments/09 nyu_sigma (notebook 06 arms: 4 and 8 blocks, 3 seeds each).
    """
    def pooled(get):
        groups = [np.array([get(r) for r in runs]) for runs in arms.values() if len(runs) > 1]
        return float(np.sqrt(sum((len(a) - 1) * a.var(ddof=1) for a in groups) / sum(len(a) - 1 for a in groups)))

    all_abs_rel = np.mean([r["nyu_depth"]["abs_rel"] for runs in arms.values() for r in runs])
    raw = dict(rmse=pooled(lambda r: r["nyu_depth"]["rmse"]), miou=pooled(lambda r: r["nyu_seg"]["mIoU"]),
               abs_rel_cv=pooled(lambda r: r["nyu_depth"]["abs_rel"]) / all_abs_rel)
    sigma = {k: max(v, SIGMA_FLOOR[k]) for k, v in raw.items()}
    sigma["raw"] = raw
    return sigma


def main_metrics(dataset: str, R: dict) -> Tuple[Optional[float], Optional[float]]:
    """(depth main metric, lower is better; segmentation mIoU) of an evaluation dict."""
    if dataset == "nyu":
        return R["nyu_depth"]["rmse"], R["nyu_seg"]["mIoU"]
    return R["cs_depth_512x1024"]["abs_rel"], R["cs_seg"]["mIoU"]


def judge(dataset: str, R_base: dict, R_new: dict, sigma: dict) -> dict:
    """z-scores of one change (new vs base); z > 0 = better, |z| >= Z = effect."""
    bd, bs = main_metrics(dataset, R_base)
    nd, ns = main_metrics(dataset, R_new)
    sd = sigma["rmse"] if dataset == "nyu" else sigma["abs_rel_cv"] * bd
    ss = sigma["miou"]
    return dict(depth_base=bd, depth_new=nd, z_depth=(bd - nd) / sd, mIoU_base=bs, mIoU_new=ns, z_seg=(ns - bs) / ss)


# =========================================================================== comparison with the original numbers

def flat_metrics(d: dict, prefix: str = "") -> Dict[str, float]:
    """Flatten an evaluation dict to {protocol.metric: value} (numbers only)."""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flat_metrics(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{prefix}{key}"] = float(value)
    return out


def compare_with_original(new: dict, original: Optional[dict]) -> Optional[dict]:
    """Largest relative difference between the reproduced and the original metrics (common keys only)."""
    if not original:
        return None
    a, b = flat_metrics(new), flat_metrics(original)
    common = [k for k in a if k in b and not k.endswith("n_images") and not k.startswith("config")]
    if not common:
        return None
    rel = {k: abs(a[k] - b[k]) / max(abs(b[k]), 1e-9) for k in common}
    worst = max(rel, key=rel.get)
    return dict(n_metrics=len(common), max_rel_diff=rel[worst], worst=worst, new=a[worst], original=b[worst])
