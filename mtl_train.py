"""Training code of the project: critic, losses, patch mixing, optimizer, schedule and the training loop.
"""
from mtl_common import *

# =========================================================================== critic (WGAN-GP, ported from SwinMTL)

def normal_init(module: nn.Module, std: float = 0.001, bias: float = 0.0) -> None:
    nn.init.normal_(module.weight, 0, std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Critic(nn.Module):
    """
    Port of SwinMTL/models/modelMulti.py::Critc, input = (depth, segmentation) map.
    """

    def __init__(self, in_channels: int = 2):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 2, kernel_size=4, stride=1, padding=1)
        )

        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def critic_input(depth: Tensor, seg: Tensor, gt_depth: Tensor, gt_seg: Tensor, dataset_cfg: dict) -> Tensor:
    """Build the 2-channel (depth, segmentation) critic input.

    Args:
        depth: depth (B, 1, H, W) in metres.
        seg: class ids (B, H, W).
        gt_depth: GT depth, its holes are copied to the input so real and fake look the same there.
        gt_seg: GT labels, ignored pixels are copied as well.
        dataset_cfg: the dataset configuration (depth range, number of classes).
    """
    valid = gt_depth > 0

    # depth, in log space as in the SwinMTL paper
    d = depth.float().clamp_min(dataset_cfg["min_depth"])
    if CFG["critic_log_depth"]:
        d = torch.log(d / dataset_cfg["min_depth"]) / math.log(dataset_cfg["max_depth"] / dataset_cfg["min_depth"])
    d = torch.where(valid, d, torch.zeros_like(d))

    # class id normalized by the number of classes
    s = seg.float()[:, None] / dataset_cfg["num_classes"]
    s = torch.where((gt_seg == 255)[:, None], torch.full_like(s, -1.0), s)

    return torch.cat([d, s], dim=1)


# =========================================================================== losses, patch mixing and critic helpers

class SiLogLoss(nn.Module):
    """
    Scale-invariant log loss (port of SwinMTL/utils/criterion.py::SiLogLoss).
    """

    def __init__(self, lambd: float = 0.5, min_depth: float = 1e-3):
        super().__init__()
        self.lambd = lambd
        self.min_depth = min_depth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        valid_mask = target > 0
        diff_log = torch.log(pred[valid_mask].clamp_min(self.min_depth)) - torch.log(target[valid_mask])
        loss = diff_log.pow(2).mean() - self.lambd * diff_log.mean().pow(2)
        return torch.sqrt(loss.clamp_min(1e-8))


def mixed_data(input_rgb: Tensor, depth_gt: Tensor, seg_gt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Patch mixing augmentation (port of SwinMTL/trainMulti.py::mixed_data).

    Args:
        input_rgb: images (B, 3, H, W).
        depth_gt: depth (B, 1, H, W).
        seg_gt: labels (B, H, W).
    """
    bs, _, h, w = input_rgb.shape
    shuffled_idx = torch.randperm(bs, device=input_rgb.device)

    # True = keep the original pixel, False = take it from the shuffled image
    filled_mask = torch.ones((bs, 1, h, w), dtype=torch.bool, device=input_rgb.device)

    for i in range(bs):
        num = torch.randint(1, 5, (1,)).item()
        size = torch.randint(h // 8, h // 2, (num,))
        x = torch.randint(0, h, (num,))
        y = torch.randint(0, w, (num,))
        for j in range(num):
            filled_mask[i, :, x[j]:x[j] + size[j], y[j]:y[j] + size[j]] = False

    input_rgb = torch.where(filled_mask, input_rgb, input_rgb[shuffled_idx])
    depth_gt = torch.where(filled_mask, depth_gt, depth_gt[shuffled_idx])
    seg_gt = torch.where(filled_mask[:, 0], seg_gt, seg_gt[shuffled_idx])
    return input_rgb, depth_gt, seg_gt


def gradient_penalty(critic: nn.Module, real: Tensor, fake: Tensor, epsilon: Tensor) -> Tensor:
    """
    WGAN-GP penalty (port of SwinMTL/trainMulti.py::get_gradient and gradient_penalty).
    """
    mixed_images = (real * epsilon + fake * (1 - epsilon)).requires_grad_(True)
    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient_norm = gradient.flatten(1).norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(flag)


# =========================================================================== optimizer and learning-rate schedule

def split_decay(named_params) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Split the trainable parameters into (with weight decay, without weight decay: biases, norms, LayerScale).
    """
    decay = []
    no_decay = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


def build_optimizer(model: nn.Module, variant_cfg: dict) -> torch.optim.Optimizer:
    """AdamW with one group per learning rate.

    Args:
        model: the model.
        variant_cfg: the variant configuration (trainable blocks, backbone LR ratio, layer decay).
    """
    param_groups = []

    # decoder + heads
    head_params = [(name, p) for name, p in model.named_parameters() if not name.startswith("backbone.")]
    decay, no_decay = split_decay(head_params)
    param_groups.append(dict(params=decay, lr=CFG["head_lr"], weight_decay=CFG["head_wd"], name="head"))
    param_groups.append(dict(params=no_decay, lr=CFG["head_lr"], weight_decay=0.0, name="head_nd"))

    # trainable ViT blocks (variant B): smaller LR, layer-wise decay
    if variant_cfg["n_trainable_blocks"] > 0:
        backbone = model.backbone
        backbone_lr = CFG["head_lr"] * variant_cfg["backbone_lr_ratio"]

        for i in range(backbone.first_trainable, backbone.n_blocks):
            block_lr = backbone_lr * variant_cfg["layer_decay"] ** (backbone.n_blocks - 1 - i)
            decay, no_decay = split_decay(backbone.vit.blocks[i].named_parameters())
            param_groups.append(dict(params=decay, lr=block_lr, weight_decay=variant_cfg["backbone_wd"], name=f"block{i}"))
            param_groups.append(dict(params=no_decay, lr=block_lr, weight_decay=0.0, name=f"block{i}_nd"))

        param_groups.append(dict(params=list(backbone.vit.norm.parameters()), lr=backbone_lr, weight_decay=0.0, name="vit_norm"))

    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Optimizer, total_iters: int):
    """
    Linear warm-up, then polynomial decay (1 - t/T)^0.9, floored at min_lr_ratio.
    """
    warmup_iters = CFG["warmup_iters"]

    def lr_factor(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return max(CFG["min_lr_ratio"], (1 - progress) ** CFG["poly_power"])

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# =========================================================================== training loop

def infinite_loader(dataset: Dataset, batch_size: int):
    """
    Yield batches forever, reshuffling at every epoch.
    """
    assert len(dataset) >= batch_size, "dataset smaller than the batch"

    while True:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=CFG["num_workers"],
            drop_last=True,
            pin_memory=device.type == "cuda",
            worker_init_fn=worker_init
        )
        yield from data_loader


def atomic_save(obj, path: Path) -> None:
    """
    Save to a temporary file and rename it, so an interruption never leaves a broken checkpoint.
    """
    tmp_path = Path(str(path) + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train_run(dataset: str, variant: str) -> dict:
    """Train one model with the unified recipe; resumes from ckpt/<run>/last.pt.

    Args:
        dataset: "cityscapes" or "nyu".
        variant: "frozen" or "partial_ft".

    Returns:
        the training history.
    """
    name = run_name(dataset, variant)
    dataset_cfg = CFG["datasets"][dataset]
    variant_cfg = CFG["variants"][variant]

    ckpt_dir = CFG["ckpt_root"] / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = CFG["results_root"] / "train" / f"{name}.json"

    if (ckpt_dir / "final.pt").exists():
        print(f"{name}: already trained")
        return load_json(history_path)

    fix_random(CFG["seed"])

    # model, critic, optimizers
    model = build_model(dataset, variant)
    model.to(device)

    critic = None
    optimizer_critic = None
    if CFG["use_critic"]:
        critic = Critic()
        critic.to(device)
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=CFG["critic_lr"], betas=(0.9, 0.999), weight_decay=CFG["critic_wd"])

    optimizer = build_optimizer(model, variant_cfg)
    total_iters = dataset_cfg["iters"]
    scheduler = build_scheduler(optimizer, total_iters)

    # train metrics init
    history = {
        "iter": [],
        "loss": [],
        "depth": [],
        "seg": [],
        "gen": [],
        "critic": [],
        "lr": [],
        "val": []
    }
    start_iter = 0

    # resume
    if (ckpt_dir / "last.pt").exists():
        ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        if critic:
            critic.load_state_dict(ckpt["critic"])
            optimizer_critic.load_state_dict(ckpt["opt_c"])
        start_iter = ckpt["iter"]
        history = ckpt["hist"]
        print(f"{name}: resumed at iteration {start_iter}")

    stats = torchinfo.summary(model, verbose=0)   # parameter counts only
    total_params, trainable_params = stats.total_params / 1e6, stats.trainable_params / 1e6
    print(f"{name}: {total_params:.1f}M params, {trainable_params:.1f}M trainable, {total_iters} iterations")

    writer = SummaryWriter(str(CFG["results_root"] / "tb" / name))
    batches = infinite_loader(make_dataset(dataset, "train", train=True), CFG["batch_size"])
    criterion_depth = SiLogLoss(0.5, dataset_cfg["min_depth"])
    criterion_seg = nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    if critic:
        critic.train()

    running = defaultdict(float)
    n_running = 0

    # train loop
    progress_bar = tqdm(range(start_iter, total_iters), initial=start_iter, total=total_iters, desc=name)
    for it in progress_bar:
        batch = next(batches)
        input_rgb = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        seg_gt = batch["seg"].to(device, non_blocking=True)

        # patch mixing augmentation
        if CFG["use_patch_mix"]:
            input_rgb, depth_gt, seg_gt = mixed_data(input_rgb, depth_gt, seg_gt)

        # forward (bf16), losses in fp32
        # frame-consistent position embeddings (ablation AB1, notebook 05); absent in the main runs
        pe_window = batch["pe_window"].to(device) if "pe_window" in batch else None
        with autocast():
            preds = model(input_rgb, pe_window=pe_window)
        pred_depth = preds["depth"].float()
        pred_seg = preds["seg"].float()

        loss_depth = criterion_depth(pred_depth, depth_gt)
        loss_seg = criterion_seg(pred_seg, seg_gt) + CFG["w_aux"] * criterion_seg(preds["aux"].float(), seg_gt)
        loss_gen = torch.zeros((), device=device)
        loss_critic = torch.zeros((), device=device)

        if critic:
            fake_depth_seg = critic_input(pred_depth, pred_seg.argmax(1), depth_gt, seg_gt, dataset_cfg)
            true_depth_seg = critic_input(depth_gt, seg_gt, depth_gt, seg_gt, dataset_cfg)

            # critic step (WGAN-GP)
            set_requires_grad(critic, True)
            optimizer_critic.zero_grad(set_to_none=True)
            fake_detached = fake_depth_seg.detach()
            epsilon = torch.rand(true_depth_seg.shape[0], 1, 1, 1, device=device)

            crit_real_pred = critic(true_depth_seg).mean()
            crit_fake_pred = critic(fake_detached).mean()
            gp = gradient_penalty(critic, true_depth_seg, fake_detached, epsilon)
            loss_critic = -(crit_real_pred - crit_fake_pred) + CFG["gp_weight"] * gp

            loss_critic.backward()
            optimizer_critic.step()

            # generator term, critic frozen
            set_requires_grad(critic, False)
            loss_gen = -critic(fake_depth_seg).mean()

        total_loss = CFG["w_depth"] * loss_depth + CFG["w_seg"] * loss_seg + CFG["w_gen"] * loss_gen

        # steps
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # metric computation and storing
        running["loss"] += total_loss.item()
        running["depth"] += loss_depth.item()
        running["seg"] += loss_seg.item()
        running["gen"] += loss_gen.item()
        running["critic"] += loss_critic.item()
        n_running += 1
        step = it + 1

        # logging
        if step % CFG["log_every"] == 0 or step == total_iters:
            averages = {key: value / n_running for key, value in running.items()}
            history["iter"].append(step)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            for key, value in averages.items():
                history[key].append(value)
                writer.add_scalar(f"train/{key}", value, step)
            writer.add_scalar("train/lr_head", optimizer.param_groups[0]["lr"], step)

            progress_bar.set_postfix(loss=f"{averages['loss']:.3f}", depth=f"{averages['depth']:.3f}", seg=f"{averages['seg']:.3f}")
            running = defaultdict(float)
            n_running = 0

        # quick validation
        if step % CFG["val_every"] == 0 or step == total_iters:
            val_metrics = quick_val(model, dataset, CFG["val_images"])
            history["val"].append(dict(iter=step, **val_metrics))
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, step)

            print(f"Iter: {step} - Val. RMSE: {val_metrics['rmse']:.4f}, AbsRel: {val_metrics['abs_rel']:.4f}, mIoU: {val_metrics['mIoU']:.2f}")
            model.train()

        # checkpoint
        if step % CFG["ckpt_every"] == 0 and step < total_iters:
            atomic_save(
                dict(
                    model=model.state_dict(),
                    opt=optimizer.state_dict(),
                    sched=scheduler.state_dict(),
                    critic=critic.state_dict() if critic else None,
                    opt_c=optimizer_critic.state_dict() if critic else None,
                    iter=step,
                    hist=history
                ),
                ckpt_dir / "last.pt"
            )

    # final weights and history
    atomic_save(dict(model=model.state_dict(), dataset=dataset, variant=variant), ckpt_dir / "final.pt")
    save_json(history, history_path)
    writer.close()

    del model, critic, optimizer, optimizer_critic
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history
