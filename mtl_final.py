"""Final-model helpers: seed noise sigma, the decision rule, and training + evaluation of the final model's runs.
"""
from mtl_train import *

FINAL_DIR = ROOT / "results" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


Z = 2.5
SIGMA_FLOOR = dict(rmse=0.003, miou=0.2, abs_rel_cv=0.01)
DECISIONS = []


def nyu_sigma() -> dict:
    """Pooled seed std over the two arms of notebook 06 (NYU, critic off), with floors."""
    path = ROOT / "results" / "compare" / "seeds_nyu_per_run.csv"
    if not path.exists():
        print("notebook 06 results not available: using the floors as sigma")
        return dict(rmse=SIGMA_FLOOR["rmse"], miou=SIGMA_FLOOR["miou"], abs_rel_cv=SIGMA_FLOOR["abs_rel_cv"], source="floors")
    df = pd.read_csv(path)

    def pooled(col):
        groups = [g[col].to_numpy() for _, g in df.groupby("arm") if len(g) > 1]
        return float(np.sqrt(sum((len(a) - 1) * a.var(ddof=1) for a in groups) / sum(len(a) - 1 for a in groups)))

    raw = dict(rmse=pooled("RMSE"), miou=pooled("mIoU"), abs_rel_cv=pooled("AbsRel") / df["AbsRel"].mean())
    sig = {k: max(v, SIGMA_FLOOR[k]) for k, v in raw.items()}
    sig["source"] = f"notebook 06 (raw: {', '.join(f'{k}={v:.4g}' for k, v in raw.items())})"
    return sig


SIGMA = nyu_sigma()
print("sigma:", SIGMA)


def main_metrics(dataset: str, R: dict) -> Tuple[float, float]:
    """(depth main metric, lower is better; segmentation mIoU) of an evaluation dict."""
    if dataset == "nyu":
        return R["nyu_depth"]["rmse"], R["nyu_seg"]["mIoU"]
    return R["cs_depth_512x1024"]["abs_rel"], R["cs_seg"]["mIoU"]


def sigmas(dataset: str, base_depth: float) -> Tuple[float, float]:
    if dataset == "nyu":
        return SIGMA["rmse"], SIGMA["miou"]
    return SIGMA["abs_rel_cv"] * base_depth, SIGMA["miou"]


def judge(name: str, dataset: str, R_base: dict, R_new: dict, kind: str = "standard") -> dict:
    """Apply the decision rule to one change (new vs base) on one dataset."""
    bd, bs = main_metrics(dataset, R_base)
    nd, ns = main_metrics(dataset, R_new)
    sd, ss = sigmas(dataset, bd)
    z_depth, z_seg = (bd - nd) / sd, (ns - bs) / ss        # > 0 = better
    better = z_depth >= Z or z_seg >= Z
    harm = z_depth <= -Z or z_seg <= -Z
    return dict(change=name, dataset=dataset, base_depth=bd, new_depth=nd, base_mIoU=bs, new_mIoU=ns,
                z_depth=z_depth, z_seg=z_seg, better=bool(better), harm=bool(harm), kind=kind)


def record(name: str, adopted, reason: str, evidence: Optional[list] = None) -> None:
    DECISIONS.append(dict(decision=name, adopted=adopted, reason=reason, evidence=evidence or []))
    print(f"[decision] {name}: {adopted}  ({reason})")
    save_json(DECISIONS, FINAL_DIR / "decisions.json")


@torch.no_grad()
def log_bias(model: nn.Module, dataset: str) -> float:
    """Mean log ratio prediction / GT with the standard single-pass protocol."""
    dcfg = CFG["datasets"][dataset]
    ds = make_dataset(dataset, dcfg["eval_split"], train=False)
    n = len(ds)
    out = []
    for i in range(n):
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


def setup_final(dataset: str, seed: int, tag: str, cfg: dict, **extra) -> str:
    """configure() for one run of the final model (or of an ablation on top of it). Returns the variant name."""
    over = dict(use_critic=cfg["use_critic"], seed=seed, frame_pe=cfg["frame_pe"], photo_rng_fix=cfg["photo_rng_fix"],
                depth_param=cfg["depth_param"])
    if dataset == "nyu":
        over.update(scale_aug=tuple(cfg["nyu_scale_aug"]), rotate_deg=cfg["nyu_rotate_deg"])
    over.update(extra)
    folder = f"{dataset}_{tag}_s{seed}"
    configure(ckpt_root=ROOT / "ckpt" / "final" / folder,
              results_root=ROOT / "results" / "final" / folder, **over)
    if dataset == "nyu" and cfg.get("nyu_crop"):
        CFG["datasets"]["nyu"]["crop"] = tuple(cfg["nyu_crop"])
    CFG["variants"]["final"] = dict(CFG["variants"]["partial_ft"], n_trainable_blocks=cfg["blocks"])
    return "final"


def train_eval(dataset: str, seed: int, tag: str, cfg: dict, **extra) -> dict:
    """Train (or resume / skip) one run and evaluate it with all protocols of notebook 03 (+ log bias). Cached."""
    variant = setup_final(dataset, seed, tag, cfg, **extra)
    print(f"=== {dataset} {tag} seed {seed}: {CFG['ckpt_root']}")
    train_run(dataset, variant)
    out_path = CFG["results_root"] / "eval" / f"{run_name(dataset, variant)}.json"
    if out_path.exists():
        return load_json(out_path)
    model = load_trained(dataset, variant)
    if dataset == "cityscapes":
        R = eval_cityscapes(model, ms=True, slide=True)
    else:
        R = eval_nyu(model, slide=True)
    R["log_bias"] = log_bias(model, dataset)
    R["config"] = dict(cfg, seed=seed, tag=tag, extra={k: str(v) for k, v in extra.items()})
    save_json(R, out_path)
    del model
    torch.cuda.empty_cache()
    return R


def load_eval(results_root: Path, name: str) -> Optional[dict]:
    path = results_root / "eval" / f"{name}.json"
    return load_json(path) if path.exists() else None
