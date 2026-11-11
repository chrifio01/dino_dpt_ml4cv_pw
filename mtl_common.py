"""
Shared code of the DINOv2 + DPT multi-task project.
"""

from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import os
import sys
import json
import math
import time
import random
import re
import copy
import zipfile
import tarfile
import warnings
import subprocess

import numpy as np
import pandas as pd
import cv2
import h5py
import scipy.io
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torchinfo
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
import albumentations as A

cv2.setNumThreads(0)  # avoid OpenCV threads inside every DataLoader worker

# Python >= 3.14 switched the Linux default start method to "forkserver", which re-imports __main__
# in every DataLoader worker and crashed the workers.
if sys.platform.startswith("linux"):
    torch.multiprocessing.set_start_method("fork", force=True)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", message="xFormers is not available")

ROOT = Path(__file__).resolve().parent
for repo in ["dinov2", "DPT", "SwinMTL"]:
    if not (ROOT / repo).exists():
        raise FileNotFoundError(f"mtl_common.py must sit next to {repo}/")

DATA_ROOT = Path( ROOT / "data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== configuration

_DEFAULT_CFG = dict(
    data_root=DATA_ROOT,
    ckpt_root=ROOT / "ckpt",
    results_root=ROOT / "results",
    seed=42,
    amp=True,
    amp_dtype=torch.bfloat16,
    num_workers=8,

    # model
    backbone="dinov2_vits14",
    layers=(2, 5, 8, 11),
    features=256,
    post_channels=(48, 96, 192, 384),
    variants={
        "frozen": dict(n_trainable_blocks=0),
        "partial_ft": dict(
            n_trainable_blocks=4,
            backbone_lr_ratio=0.1,
            layer_decay=0.9,
            backbone_wd=0.05,
            grad_ckpt=True
        ),
    },

    # unified training recipe
    batch_size=8,
    head_lr=1e-4,
    head_wd=1e-2,
    warmup_iters=1500,
    poly_power=0.9,
    min_lr_ratio=0.01,
    w_depth=1.0,
    w_seg=1.0,
    w_aux=0.4,
    w_gen=0.1,
    use_critic=True,
    critic_lr=1e-4,
    critic_wd=1e-4,
    gp_weight=100.0,
    critic_log_depth=True,
    use_patch_mix=True,
    frame_pe=False,     # ablation AB1: frame-consistent position embeddings (notebook 05)
    scale_aug=(1.0, 1.5),
    photo_rng_fix=False,  # ablation AUG-D: seed the photometric augmentation per DataLoader worker
    rotate_deg=0.0,     # ablation AUG-C: random rotation in [-rotate_deg, rotate_deg] degrees (0 = off, no random draw)
    depth_param="linear",     # "linear": sigmoid * max_depth; "log": SwinMTL's log-space output (ablation LOG, notebook 09)
    depth_head="regression",  # "regression" or "bins": DINOv2's 256 uniform bins with linear normalisation (ablation D4)
    n_depth_bins=256,         # number of bins of the "bins" depth head
    log_every=50,
    ckpt_every=2000,
    val_every=4000,
    val_images=100,

    # datasets
    datasets={
        "cityscapes": dict(
            num_classes=20,
            eval_classes=19,
            void_class=19,
            min_depth=1e-3,
            max_depth=80.0,
            base_size=(512, 1024),
            crop=(448, 896),
            iters=40000,
            fx_base=2262.52 * 0.5,
            eval_split="val"
        ),
        "nyu": dict(
            num_classes=40,
            eval_classes=40,
            void_class=None,
            min_depth=1e-3,
            max_depth=10.0,
            base_size=(480, 640),
            crop=(476, 630),
            iters=20000,
            fx_base=518.86,
            eval_split="test"
        ),
    },

    crestereo=dict(full_res=True, iters=20, gdrive_id="1D2s1v4VhJlNz98FQpFxf_kBAKQVN_7xo"),
    zero_shot=dict(kitti=True, sunrgbd=True, ade20k=True, voc2012=True, pcontext=True),
    ms_scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
)

CFG = {}


def configure(**overrides) -> dict:
    """Build the global configuration CFG from the defaults.

    Args:
        overrides: any configuration key to replace, e.g. batch_size=4.
    """
    CFG.clear()
    CFG.update(copy.deepcopy(_DEFAULT_CFG))
    CFG.update(**overrides)

    for key in ["data_root", "ckpt_root", "results_root"]:
        CFG[key].mkdir(parents=True, exist_ok=True)

    fix_random(CFG["seed"])
    return CFG


RUNS = [("cityscapes", "frozen"), ("cityscapes", "partial_ft"), ("nyu", "frozen"), ("nyu", "partial_ft")]
VARIANT_LABEL = {"frozen": "Mine-A (frozen)", "partial_ft": "Mine-B (partial FT)"}

# dataset locations
CS_ROOT = DATA_ROOT / "cityscapes"
CS_DEPTH_DIR = CS_ROOT / "crestereo_depth"
NYU_ROOT = DATA_ROOT / "nyu"
KITTI_ROOT = DATA_ROOT / "kitti"
SUN_ROOT = DATA_ROOT / "sunrgbd"
ADE_ROOT = DATA_ROOT / "ade20k"
VOC_ROOT = DATA_ROOT / "voc"
PC_ROOT = DATA_ROOT / "pcontext"


# =========================================================================== utilities

# DINOv2 was pretrained with the ImageNet statistics
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init(worker_id: int) -> None:
    """
    Give every DataLoader worker its own numpy seed (numpy is not reseeded by PyTorch).
    """
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
    # ablation AUG-D: albumentations 2.x keeps its own RNG, which fix_random does not seed. Forked workers
    # inherit an identical copy of it, so without this all workers draw the same photometric parameters, and the
    # same sequence repeats every epoch. Off by default so that earlier runs stay reproducible as they were.
    if CFG.get("photo_rng_fix", False):
        PHOTO_AUG.set_random_seed((torch.initial_seed() + worker_id) % 2**32)


def to_tensor(img_rgb_uint8: np.ndarray) -> Tensor:
    """
    Convert an RGB uint8 image (H, W, 3) into a normalized float tensor (3, H, W).
    """
    img = img_rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


def denormalize(x: Tensor) -> np.ndarray:
    """
    Convert a normalized tensor (3, H, W) back into an image (H, W, 3) in [0, 1], for plotting.
    """
    img = x.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMNET_STD + IMNET_MEAN
    return np.clip(img, 0, 1)


def prep_image(img_rgb_uint8: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tensor:
    """Prepare a numpy RGB image as model input.

    Args:
        img_rgb_uint8: the image (H, W, 3).
        size: optional output size (h, w).

    Returns:
        a normalized tensor (1, 3, h, w) on the device.
    """
    if size is not None and tuple(size) != img_rgb_uint8.shape[:2]:
        h, w = size
        # area interpolation when shrinking, bilinear when enlarging
        interpolation = cv2.INTER_AREA if h < img_rgb_uint8.shape[0] else cv2.INTER_LINEAR
        img_rgb_uint8 = cv2.resize(img_rgb_uint8, (w, h), interpolation=interpolation)

    return to_tensor(img_rgb_uint8)[None].to(device)


def autocast():
    """
    Mixed precision context (bf16 by default).
    """
    return torch.autocast(device_type=device.type, dtype=CFG["amp_dtype"], enabled=CFG["amp"])


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def download(url: str, dst, session=None, chunk: int = 1 << 20) -> Path:
    """Stream a file to disk, skipping it if it already exists.

    Args:
        url: the file URL.
        dst: the destination path.
        session: optional requests session (e.g. logged in to Cityscapes).
        chunk: chunk size in bytes.
    """
    dst = Path(dst)
    if dst.exists():
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")  # renamed only when the download is complete

    with (session or requests).get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name, leave=False) as bar:
            for block in response.iter_content(chunk):
                f.write(block)
                bar.update(len(block))

    tmp.rename(dst)
    return dst


def extract(archive, dst, keep=None) -> None:
    """Extract a zip or tar archive.

    Args:
        archive: the archive path.
        dst: the destination folder.
        keep: optional predicate on the member name, to unpack only what is needed.
    """
    archive = Path(archive)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if keep is None or keep(m)]
            for member in tqdm(members, desc=f"unzip {archive.name}", leave=False):
                z.extract(member, dst)
    else:
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if keep is None or keep(m.name)]
            t.extractall(dst, members=members, filter="data")


# =========================================================================== label spaces

from cityscapesscripts.helpers.labels import labels as CS_LABELS

# Cityscapes: labelId -> trainId, every ignored label becomes void = 19 (as in SwinMTL/dataset/labels.py)
CS_ID2TRAIN = np.full(256, 19, np.uint8)
for label in CS_LABELS:
    if 0 <= label.id < 256:
        CS_ID2TRAIN[label.id] = label.trainId if 0 <= label.trainId < 19 else 19

cs_train_labels = sorted([l for l in CS_LABELS if 0 <= l.trainId < 19], key=lambda l: l.trainId)
CS_NAMES = [l.name for l in cs_train_labels] + ["void"]
CS_PALETTE = np.array([l.color for l in cs_train_labels] + [(0, 0, 0)], np.uint8)

NYU40_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel",
    "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop"
]

# 20 tab20 colours, darkened for the second 20 classes
nyu_colors = plt.get_cmap("tab20")(np.arange(40) % 20)[:, :3] * 255
nyu_shade = np.where(np.arange(40) < 20, 1.0, 0.6)[:, None]
NYU_PALETTE = (nyu_colors * nyu_shade).astype(np.uint8)

ADE150_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence",
    "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]

VOC_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

PC59_NAMES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book", "bottle",
    "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth", "computer", "cow", "cup", "curtain",
    "dog", "door", "fence", "floor", "flower", "food", "grass", "ground", "horse", "keyboard", "light",
    "motorbike", "mountain", "mouse", "person", "plate", "platform", "pottedplant", "road", "rock", "sheep",
    "shelves", "sidewalk", "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood"
]

assert (len(CS_NAMES), len(NYU40_NAMES), len(ADE150_NAMES), len(VOC_NAMES), len(PC59_NAMES)) == (20, 40, 150, 21, 59)

SOURCE_NAMES = {"cityscapes": CS_NAMES, "nyu": NYU40_NAMES}
PALETTES = {"cityscapes": CS_PALETTE, "nyu": NYU_PALETTE}


# =========================================================================== augmentation

# SwinMTL basic_transform (image only)
PHOTO_AUG = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.HueSaturationValue(p=0.5)
])


def geometric_aug(
        img: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        crop: Tuple[int, int],
        scale_range: Tuple[float, float],
        seg_fill: int = 255,
        return_window: bool = False,
        rotate_deg: float = 0.0
        ):
    """Apply the same random rescale, crop and flip to image, depth and segmentation.

    Args:
        img: RGB image (H, W, 3).
        depth: metric depth (H, W), 0 = invalid.
        seg: segmentation labels (H, W).
        crop: output size (h, w).
        scale_range: range of the random rescale factor s.
        seg_fill: label used if padding is needed.
        return_window: also return (y, x, frame_h, frame_w), the crop's position inside the rescaled frame
            (mirrored if flipped). Used by the frame-consistent position embeddings (AB1).
        rotate_deg: if > 0, rotate by a random angle in [-rotate_deg, rotate_deg] after the rescale (ablation AUG-C).
            The corners left uncovered become invalid depth (0) and ignored labels (seg_fill). With 0, no random
            number is drawn, so the random stream is the same as before this option existed.
    """
    # random rescale: depth is divided by s to stay geometrically consistent
    s = np.random.uniform(*scale_range)
    if abs(s - 1.0) > 1e-3:
        H, W = img.shape[:2]
        new_h, new_w = int(round(H * s)), int(round(W * s))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST) / np.float32(s)
        seg = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # random rotation about the image centre (BTS-style NYU depth augmentation)
    if rotate_deg > 0:
        angle = np.random.uniform(-rotate_deg, rotate_deg)
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        seg = cv2.warpAffine(seg, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=seg_fill)

    # pad if the image is smaller than the crop
    crop_h, crop_w = crop
    H, W = img.shape[:2]
    if H < crop_h or W < crop_w:
        pad_h, pad_w = max(crop_h - H, 0), max(crop_w - W, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
        depth = np.pad(depth, ((0, pad_h), (0, pad_w)))
        seg = np.pad(seg, ((0, pad_h), (0, pad_w)), constant_values=seg_fill)
        H, W = img.shape[:2]

    # random crop
    y = np.random.randint(0, H - crop_h + 1)
    x = np.random.randint(0, W - crop_w + 1)
    img = img[y:y + crop_h, x:x + crop_w]
    depth = depth[y:y + crop_h, x:x + crop_w]
    seg = seg[y:y + crop_h, x:x + crop_w]

    # horizontal flip
    flipped = np.random.rand() < 0.5
    if flipped:
        img = img[:, ::-1]
        depth = depth[:, ::-1]
        seg = seg[:, ::-1]

    out = (np.ascontiguousarray(img), np.ascontiguousarray(depth), np.ascontiguousarray(seg))
    if return_window:
        window = (y, W - x - crop_w if flipped else x, H, W)
        return out + (window,)
    return out


# =========================================================================== datasets

class MTLDataset(Dataset):
    """
    Base class of the multi-task datasets: subclasses only implement load(i).
    """

    def __init__(self, name: str, split: str, train: bool) -> None:
        super().__init__()

        self.name = name
        self.split = split
        self.train = train
        self.dc = CFG["datasets"][name]
        self.items = []

    def __len__(self) -> int:
        return len(self.items)

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the RGB uint8 image, the metric depth (0 = invalid) and the segmentation labels.
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict:
        img, depth, seg = self.load(i)
        base_h, base_w = self.dc["base_size"]

        if self.train:
            # resize to the base resolution
            if img.shape[:2] != (base_h, base_w):
                img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (base_w, base_h), interpolation=cv2.INTER_NEAREST)
                seg = cv2.resize(seg, (base_w, base_h), interpolation=cv2.INTER_NEAREST)

            # geometric + photometric augmentation
            img, depth, seg, window = geometric_aug(img, depth, seg, self.dc["crop"], CFG["scale_aug"], return_window=True,
                                                     rotate_deg=CFG.get("rotate_deg", 0.0))
            img = PHOTO_AUG(image=img)["image"]

            sample = dict(
                image=to_tensor(img),
                depth=torch.from_numpy(depth.astype(np.float32))[None],
                seg=torch.from_numpy(seg.astype(np.int64))
            )
            if CFG.get("frame_pe", False):
                sample["pe_window"] = torch.tensor(window, dtype=torch.float32)
            return sample

        # eval: image at the base resolution, ground truth at full resolution
        if img.shape[:2] == (base_h, base_w):
            small = img
        else:
            small = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_AREA)

        return dict(
            image=to_tensor(small),
            depth=torch.from_numpy(depth.astype(np.float32)),
            seg=torch.from_numpy(seg.astype(np.int64)),
            idx=i
        )


class CityscapesMTL(MTLDataset):
    def __init__(self, split: str, train: bool) -> None:
        super().__init__("cityscapes", split, train)
        self.items = sorted((CS_ROOT / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_path = self.items[i]
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        label_ids = cv2.imread(str(CS_ROOT / "gtFine" / self.split / city / f"{stem}_gtFine_labelIds.png"), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(CS_DEPTH_DIR / self.split / city / f"{stem}_crestereo_depth.png"), cv2.IMREAD_UNCHANGED)

        depth = depth.astype(np.float32) / 256.0  # stored as metres x 256
        seg = CS_ID2TRAIN[label_ids]
        return img, depth, seg


class NYUMTL(MTLDataset):
    def __init__(self, split: str, train: bool) -> None:
        super().__init__("nyu", split, train)
        self.items = sorted((NYU_ROOT / split / "rgb").glob("*.png"))

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_path = self.items[i]

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(NYU_ROOT / self.split / "depth" / img_path.name), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(NYU_ROOT / self.split / "seg" / img_path.name), cv2.IMREAD_UNCHANGED)

        depth = depth.astype(np.float32) / 1000.0  # stored in millimetres
        # 0 (unlabelled) -> 255 ignored, 1..40 -> 0..39
        label = label.astype(np.int64)
        seg = np.where(label == 0, 255, label - 1).astype(np.uint8)
        return img, depth, seg


def make_dataset(name: str, split: str, train: bool) -> Dataset:
    """
    Single factory used by training and evaluation.
    """
    if name == "cityscapes":
        return CityscapesMTL(split, train)
    return NYUMTL(split, train)


class KittiEigenTest(Dataset):
    """
    KITTI Eigen test split: returns (rgb, depth_gt [m], focal).
    The paths in eigen_test.txt are relative to KITTI_ROOT, so the folder can be moved.
    """

    def __init__(self) -> None:
        super().__init__()
        lines = (KITTI_ROOT / "eigen_test.txt").read_text().splitlines()
        self.items = [line.split() for line in lines]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path, gt_path, focal = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(KITTI_ROOT / img_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(KITTI_ROOT / gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        return rgb, depth, float(focal)


class SunRGBDTest(Dataset):
    """
    SUN RGB-D test split: returns (rgb, depth_gt [m], fx).
    """

    def __init__(self) -> None:
        super().__init__()
        scenes = (SUN_ROOT / "test_scenes.txt").read_text().splitlines()
        self.items = [SUN_ROOT / "SUNRGBD" / scene for scene in scenes]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        scene = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(next((scene / "image").glob("*.jpg")))), cv2.COLOR_BGR2RGB)

        # toolbox encoding: circular bit shift, then millimetres
        raw = cv2.imread(str(next((scene / "depth_bfx").glob("*.png"))), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        depth = ((raw >> 3) | (raw << 13)).astype(np.float32) / 1000.0

        K = np.loadtxt(scene / "intrinsics.txt").reshape(3, 3)
        return rgb, depth, float(K[0, 0])


class ADE20KVal(Dataset):
    """
    ADE20K val: returns (rgb, gt) with gt = value - 1 and "other" -> 255.
    """

    def __init__(self) -> None:
        super().__init__()
        base = ADE_ROOT / "ADEChallengeData2016"
        self.items = sorted((base / "images" / "validation").glob("*.jpg"))
        self.ann_dir = base / "annotations" / "validation"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        gt = cv2.imread(str(self.ann_dir / (img_path.stem + ".png")), cv2.IMREAD_UNCHANGED).astype(np.int64) - 1
        gt[gt < 0] = 255
        return rgb, gt


class VOC2012Val(Dataset):
    """
    Pascal VOC 2012 val: returns (rgb, gt) with palette indices, 255 = border.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base = VOC_ROOT / "VOCdevkit" / "VOC2012"
        self.items = (self.base / "ImageSets" / "Segmentation" / "val.txt").read_text().split()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        name = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(self.base / "JPEGImages" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        gt = np.array(Image.open(self.base / "SegmentationClass" / f"{name}.png")).astype(np.int64)
        return rgb, gt


class PContextVal(Dataset):
    """
    Pascal Context val: the 459-class LabelMap is converted to PC-59 indices by class name.
    The images are the VOC 2010 ones, read from VOC 2012 (which contains all of them).
    """

    def __init__(self) -> None:
        super().__init__()
        self.base = VOC_ROOT / "VOCdevkit" / "VOC2012"

        # one .mat file per image, named like 2008_000002
        mats = {}
        for mat_path in PC_ROOT.rglob("*.mat"):
            if re.fullmatch(r"\d{4}_\d{6}", mat_path.stem):
                mats[mat_path.stem] = mat_path

        names = (PC_ROOT / "val.txt").read_text().split()
        self.items = [(name, mats[name]) for name in names if name in mats]

        # 459 ids -> PC-59 index (everything else -> 255)
        self.lut = np.full(1024, 255, np.int64)
        labels_txt = next(PC_ROOT.rglob("labels.txt"))
        for line in labels_txt.read_text().splitlines():
            match = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
            if match and match.group(2).strip() in PC59_NAMES:
                self.lut[int(match.group(1))] = PC59_NAMES.index(match.group(2).strip())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name, mat_path = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(self.base / "JPEGImages" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        label_map = scipy.io.loadmat(mat_path)["LabelMap"].astype(np.int64)
        return rgb, self.lut[label_map]


def make_eval_dataset(kind: str) -> Dataset:
    """
    Factory of the eval-only readers (kitti, sunrgbd, ade20k, voc2012, pcontext).
    """
    readers = dict(kitti=KittiEigenTest, sunrgbd=SunRGBDTest, ade20k=ADE20KVal, voc2012=VOC2012Val, pcontext=PContextVal)
    return readers[kind]()


def colorize_seg(seg, palette: np.ndarray) -> np.ndarray:
    """
    Map class ids to colours; ids outside the palette (ignore) stay black.
    """
    seg = np.asarray(seg)
    out = np.zeros((*seg.shape, 3), np.uint8)
    valid = seg < len(palette)
    out[valid] = palette[seg[valid]]
    return out


# =========================================================================== model

class DinoV2Backbone(nn.Module):

    def __init__(
            self,
            name: str = "dinov2_vits14",
            layers: Sequence[int] = (2, 5, 8, 11),   # blocks 3, 6, 9, 12
            n_trainable_blocks: int = 0,             # 0 = frozen backbone (variant A)
            grad_ckpt: bool = False,                 # gradient checkpointing on the trainable blocks
            pretrained: bool = True
            ):
        super(DinoV2Backbone, self).__init__()

        # load from the local repository (only dinov2.models is imported)
        if str(ROOT / "dinov2") not in sys.path:
            sys.path.insert(0, str(ROOT / "dinov2"))
        from dinov2.hub import backbones as dino_hub

        self.vit = getattr(dino_hub, name)(pretrained=pretrained)
        self.layers = list(layers)
        self.patch_size = self.vit.patch_size
        self.embed_dim = self.vit.embed_dim
        self.n_blocks = len(self.vit.blocks)
        self.first_trainable = self.n_blocks - n_trainable_blocks
        self.frozen = n_trainable_blocks == 0
        self.grad_ckpt = grad_ckpt

        # freeze everything, then unfreeze the last blocks (+ final norm)
        for param in self.vit.parameters():
            param.requires_grad = False

        for block in self.vit.blocks[self.first_trainable:]:
            for param in block.parameters():
                param.requires_grad = True

        if not self.frozen:
            for param in self.vit.norm.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)

        # frozen blocks are always in inference mode
        for i, block in enumerate(self.vit.blocks):
            if i < self.first_trainable:
                block.eval()

        if self.frozen:
            self.vit.eval()

        return self

    def frame_tokens(self, x: Tensor, pe_window: Tensor) -> Tensor:
        """Patch embedding + frame-consistent position embeddings (ablation AB1).

        The position-embedding table is interpolated to the token grid of the whole (rescaled) frame the
        crop was taken from, and the crop's window is sliced out of it. At test time the full frame gets the
        standard interpolation, which is then the same grid, so train and test see the same embedding spacing.

        Args:
            x: crops (B, 3, H, W).
            pe_window: (B, 4) = (y, x, frame_h, frame_w) in pixels of the rescaled frame.
        """
        p = self.patch_size
        B, _, H, W = x.shape
        h, w = H // p, W // p
        patches = self.vit.patch_embed(x)
        C = patches.shape[-1]

        cache = {}
        pos = []
        for y0, x0, frame_h, frame_w in pe_window.tolist():
            gh, gw = math.ceil(frame_h / p), math.ceil(frame_w / p)   # same grid as pad_to_multiple at test time
            if (gh, gw) not in cache:
                dummy = patches.new_zeros(1, gh * gw + 1, C)
                cache[(gh, gw)] = self.vit.interpolate_pos_encoding(dummy, gh * p, gw * p).float()
            full = cache[(gh, gw)]
            ty = min(max(int(round(y0 / p)), 0), gh - h)
            tx = min(max(int(round(x0 / p)), 0), gw - w)
            grid = full[:, 1:].reshape(1, gh, gw, C)[:, ty:ty + h, tx:tx + w].reshape(1, h * w, C)
            pos.append(torch.cat([full[:, :1], grid], dim=1))

        tokens = torch.cat([self.vit.cls_token.expand(B, -1, -1), patches], dim=1)
        tokens = tokens + torch.cat(pos, dim=0).to(tokens.dtype)
        if self.vit.register_tokens is not None:
            tokens = torch.cat([tokens[:, :1], self.vit.register_tokens.expand(B, -1, -1), tokens[:, 1:]], dim=1)
        return tokens

    def forward(self, x: Tensor, pe_window: Optional[Tensor] = None) -> List[Tuple[Tensor, Tensor]]:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # patch embedding + positional encoding (always frozen)
        with torch.no_grad():
            if pe_window is None:
                tokens = self.vit.prepare_tokens_with_masks(x)
            else:
                tokens = self.frame_tokens(x, pe_window)

        # transformer blocks
        outputs = []
        for i, block in enumerate(self.vit.blocks):
            if i < self.first_trainable:
                with torch.no_grad():
                    tokens = block(tokens)
            elif self.grad_ckpt and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)

            if i in self.layers:
                outputs.append(tokens)

        # final norm, then split CLS token and patch tokens
        n_registers = self.vit.num_register_tokens
        feats = []
        with torch.no_grad() if self.frozen else nullcontext():
            for out in outputs:
                out = self.vit.norm(out)
                patches = out[:, 1 + n_registers:].transpose(1, 2).reshape(B, -1, h, w)
                cls_token = out[:, 0]
                feats.append((patches, cls_token))

        return feats


class Reassemble(nn.Module):

    def __init__(self, embed_dim: int = 384, out_channels: Sequence[int] = (48, 96, 192, 384)):
        super(Reassemble, self).__init__()

        # readout "project": concat CLS token to every patch token, then Linear + GELU
        self.readout = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.GELU())
            for _ in out_channels
        ])

        # 1x1 projection to the pyramid channels
        self.project = nn.ModuleList([nn.Conv2d(embed_dim, c, kernel_size=1) for c in out_channels])

        # resample x4, x2, x1, x1/2
        c0, c1, _, c3 = out_channels
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(c0, c0, kernel_size=4, stride=4),
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
        ])

    def forward(self, feats: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        outputs = []

        for i, (x, cls_token) in enumerate(feats):
            B, C, h, w = x.shape

            # read
            tokens = x.flatten(2).transpose(1, 2)    # (B, hw, C)
            readout = cls_token[:, None].expand_as(tokens)
            tokens = self.readout[i](torch.cat([tokens, readout], dim=-1))

            # concatenate back into a map, project and resample
            x = tokens.transpose(1, 2).reshape(B, C, h, w)
            x = self.project[i](x)
            x = self.resample[i](x)
            outputs.append(x)

        return outputs


class ResidualConvUnit(nn.Module):

    def __init__(self, features: int, bn: bool = True):
        super(ResidualConvUnit, self).__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not bn)
        self.bn1 = nn.BatchNorm2d(features) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(features) if bn else nn.Identity()
        self.activation = nn.ReLU(False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.bn1(self.conv1(out))

        out = self.activation(out)
        out = self.bn2(self.conv2(out))

        # residual connection
        return out + x


class FeatureFusionBlock(nn.Module):

    def __init__(self, features: int, bn: bool = True):
        super(FeatureFusionBlock, self).__init__()

        self.res1 = ResidualConvUnit(features, bn)
        self.res2 = ResidualConvUnit(features, bn)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None, size=None) -> Tensor:
        # add the skip connection
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.res1(skip)

        x = self.res2(x)

        # upsample x2, or to the size of the next level
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        return self.out_conv(x)


class DinoDPT(nn.Module):

    def __init__(
            self,
            num_classes: int,
            max_depth: float,
            n_trainable_blocks: int = 0,
            min_depth: float = 1e-3,
            grad_ckpt: bool = False,
            pretrained: bool = True,
            features: Optional[int] = None,
            post_channels: Optional[Sequence[int]] = None
            ):
        super(DinoDPT, self).__init__()

        features = features or CFG["features"]
        post_channels = post_channels or CFG["post_channels"]
        self.max_depth = max_depth

        # depth output parameterisation (ablation LOG). With "log", a buffer is registered so that a
        # checkpoint can only be loaded into a model with the same parameterisation (strict loading fails otherwise).
        self.depth_param = CFG.get("depth_param", "linear")
        assert self.depth_param in ("linear", "log")
        if self.depth_param == "log":
            self.register_buffer("log_depth_range", torch.tensor([min_depth, max_depth], dtype=torch.float32))

        # encoder
        self.backbone = DinoV2Backbone(CFG["backbone"], CFG["layers"], n_trainable_blocks, grad_ckpt, pretrained)

        # shared DPT decoder
        self.reassemble = Reassemble(self.backbone.embed_dim, post_channels)
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(c, features, kernel_size=3, padding=1, bias=False)
            for c in post_channels
        ])
        self.refine = nn.ModuleList([FeatureFusionBlock(features, bn=True) for _ in range(4)])

        # depth head (DPT monodepth head)
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1 if CFG.get("depth_head", "regression") == "regression" else CFG.get("n_depth_bins", 256), kernel_size=1)
        )

        # ablation D4: classification into uniformly distributed bins (DINOv2, dinov2/hub/depth/decode_heads.py,
        # bins_strategy "UD", norm_strategy "linear" following AdaBins); depth = sum_i p_i * c_i
        self.depth_head_type = CFG.get("depth_head", "regression")
        assert self.depth_head_type in ("regression", "bins")
        if self.depth_head_type == "bins":
            assert self.depth_param == "linear", "the bins head replaces the output parameterisation"
            self.register_buffer("bin_centers", torch.linspace(min_depth, max_depth, CFG.get("n_depth_bins", 256)), persistent=False)

        # segmentation head + auxiliary head (DPT segmentation head)
        self.seg_head = self._make_seg_head(features, num_classes)
        self.aux_head = self._make_seg_head(features, num_classes)

    def _bins_to_depth(self, h: Tensor) -> Tensor:
        """D4: last conv -> 256 bin scores -> linear normalisation (relu + 0.1, / sum) -> depth = sum_i p_i * c_i."""
        logit = torch.relu(self.depth_head[-1](h).float()) + 0.1
        prob = logit / logit.sum(dim=1, keepdim=True)
        return (prob * self.bin_centers.view(1, -1, 1, 1)).sum(1, keepdim=True)

    def _bins_to_depth_ckpt(self, h: Tensor) -> Tensor:
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._bins_to_depth, h, use_reentrant=False)
        return self._bins_to_depth(h)

    @staticmethod
    def _make_seg_head(features: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(features, num_classes, kernel_size=1)
        )

    def forward(self, x: Tensor, pe_window: Optional[Tensor] = None) -> Dict[str, Tensor]:
        H, W = x.shape[-2:]

        # backbone + reassemble (pe_window: frame-consistent position embeddings, ablation AB1)
        pyramid = self.reassemble(self.backbone(x, pe_window))
        l1, l2, l3, l4 = [layer_rn(f) for layer_rn, f in zip(self.layer_rn, pyramid)]

        # fusion from coarse to fine
        p4 = self.refine[3](l4, size=l3.shape[-2:])
        p3 = self.refine[2](p4, l3, size=l2.shape[-2:])
        p2 = self.refine[1](p3, l2, size=l1.shape[-2:])
        p1 = self.refine[0](p2, l1)

        # heads, resized to the input resolution
        if self.depth_head_type == "bins":
            # bins: depth at the head's resolution first, then upsampled (no 256-channel map at full resolution)
            h = self.depth_head[:-1](p1)
            depth_small = torch.cat([self._bins_to_depth_ckpt(h[i:i + 1]) for i in range(h.shape[0])])
            depth_logits = None
        else:
            depth_logits = F.interpolate(self.depth_head(p1), size=(H, W), mode="bilinear", align_corners=False)
        seg_logits = F.interpolate(self.seg_head(p1), size=(H, W), mode="bilinear", align_corners=False)

        out = {
            "depth": self.depth_from_logits(depth_logits) if depth_logits is not None
            else F.interpolate(depth_small, size=(H, W), mode="bilinear", align_corners=False),
            "seg": seg_logits
        }

        if self.training:
            out["aux"] = F.interpolate(self.aux_head(p4), size=(H, W), mode="bilinear", align_corners=False)

        return out


def _depth_from_logits(self, logits: Tensor) -> Tensor:
    """linear: sigmoid * max_depth (DPT / DINOv2 style). log: d_min * exp(log(d_max / d_min) * sigmoid), SwinMTL's
    output (SwinMTL/models/modelMulti.py), uniform in log depth."""
    if self.depth_param == "log":
        d_min, d_max = self.log_depth_range[0], self.log_depth_range[1]
        return d_min * torch.exp(torch.log(d_max / d_min) * torch.sigmoid(logits.float()))
    return torch.sigmoid(logits) * self.max_depth


DinoDPT.depth_from_logits = _depth_from_logits


def build_model(dataset: str, variant: str, pretrained: bool = True) -> DinoDPT:
    """
    Create the model of a dataset ("cityscapes", "nyu") and a variant ("frozen", "partial_ft") from CFG.
    """
    dataset_cfg = CFG["datasets"][dataset]
    variant_cfg = CFG["variants"][variant]

    return DinoDPT(
        num_classes=dataset_cfg["num_classes"],
        max_depth=dataset_cfg["max_depth"],
        min_depth=dataset_cfg["min_depth"],
        n_trainable_blocks=variant_cfg["n_trainable_blocks"],
        grad_ckpt=variant_cfg.get("grad_ckpt", False),
        pretrained=pretrained
    )


def run_name(dataset: str, variant: str) -> str:
    return f"{dataset}_{variant}"


def load_trained(dataset: str, variant: str) -> DinoDPT:
    """
    Load the final checkpoint of a run, in eval mode.
    """
    model = build_model(dataset, variant, pretrained=False)
    ckpt = torch.load(CFG["ckpt_root"] / run_name(dataset, variant) / "final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


# =========================================================================== evaluator

def depth_metrics(pred: Tensor, gt: Tensor) -> dict:
    """
    Depth metrics of one image (port of SwinMTL/utils/metrics.py::eval_depth).
    """
    thresh = torch.maximum(gt / pred, pred / gt)
    diff = pred - gt
    diff_log = torch.log(pred) - torch.log(gt)

    return dict(
        d1=(thresh < 1.25).float().mean().item(),
        d2=(thresh < 1.25 ** 2).float().mean().item(),
        d3=(thresh < 1.25 ** 3).float().mean().item(),
        abs_rel=(diff.abs() / gt).mean().item(),
        sq_rel=(diff.pow(2) / gt).mean().item(),
        rmse=diff.pow(2).mean().sqrt().item(),
        rmse_log=diff_log.pow(2).mean().sqrt().item(),
        log10=(torch.log10(pred) - torch.log10(gt)).abs().mean().item(),
        silog=(diff_log.pow(2).mean() - 0.5 * diff_log.mean().pow(2)).clamp_min(0).sqrt().item()
    )


class MeanMeter:
    """
    Average of per-image metric dictionaries.
    """

    def __init__(self):
        self.sum = defaultdict(float)
        self.n = 0

    def update(self, metrics: dict) -> None:
        for key, value in metrics.items():
            self.sum[key] += value
        self.n += 1

    def result(self) -> dict:
        results = {key: value / max(self.n, 1) for key, value in self.sum.items()}
        results["n_images"] = self.n
        return results


class DepthMeter(MeanMeter):
    """
    Per-image depth metrics inside the valid range, averaged over the images.
    """

    def __init__(self, min_depth: float, max_depth: float, median_scale: bool = False):
        super().__init__()
        self.min = min_depth
        self.max = max_depth
        self.median_scale = median_scale  # only for the zero-shot transfers

    def update(self, pred: Tensor, gt: Tensor, mask: Optional[Tensor] = None) -> None:
        valid = (gt > self.min) & (gt < self.max)
        if mask is not None:
            valid &= mask
        if valid.sum() == 0:
            return

        pred = pred[valid].float()
        gt = gt[valid].float()

        if self.median_scale:
            pred = pred * (gt.median() / pred.median())

        # same treatment of inf / nan as SwinMTL
        pred = torch.nan_to_num(pred, nan=self.min, posinf=self.max).clamp(self.min, self.max)
        super().update(depth_metrics(pred, gt))


class SegMeter:
    """
    Dataset-level confusion matrix: mIoU, pixel accuracy and per-class IoU.
    """

    def __init__(self, num_classes: int):
        self.n = num_classes
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: Tensor, gt: Tensor) -> None:
        mask = gt != 255
        indices = gt[mask].long() * self.n + pred[mask].long()
        self.cm += torch.bincount(indices.cpu(), minlength=self.n ** 2).reshape(self.n, self.n)

    def result(self, classes=None) -> dict:
        cm = self.cm.double()
        tp = cm.diag()
        gt_total = cm.sum(1)
        pred_total = cm.sum(0)
        iou = tp / (gt_total + pred_total - tp)

        classes = list(range(self.n)) if classes is None else list(classes)
        selected_iou = iou[classes]
        present = gt_total[classes] > 0  # mean only over the classes present in the GT

        per_class_iou = []
        for value, n_gt in zip(selected_iou.tolist(), gt_total[classes].tolist()):
            per_class_iou.append(100 * value if n_gt > 0 else None)

        return dict(
            mIoU=100 * selected_iou[present].mean().item() if present.any() else float("nan"),
            pixAcc=100 * (tp.sum() / cm.sum().clamp_min(1)).item(),
            per_class_iou=per_class_iou
        )


def swinmtl_eval_seg(pred: Tensor, target: Tensor, num_classes: int) -> dict:
    """
    Per-image mIoU over the classes present in the GT (port of SwinMTL/utils/metrics.py::eval_seg).
    """
    pred = pred.flatten()
    target = target.flatten()

    valid_mask = (target != 255) & (target != -1)
    pred = pred[valid_mask]
    target = target[valid_mask]

    present = torch.unique(target)
    miou = 0.0
    for c in present.tolist():
        if c >= num_classes:
            continue

        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        miou += (tp / (tp + fp + fn + 1e-6)).item()

    return dict(
        mIoU=100 * miou / max(len(present), 1),
        pixAcc=100 * (pred == target).float().mean().item()
    )


def pad_to_multiple(x: Tensor, m: int = 14) -> Tuple[Tensor, Tuple[int, int, int, int]]:
    """
    Centre-pad to a multiple of the patch size; returns the padded tensor and (top, left, H, W) to crop back.
    """
    H, W = x.shape[-2:]
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    top = pad_h // 2
    left = pad_w // 2
    x = F.pad(x, (left, pad_w - left, top, pad_h - top))
    return x, (top, left, H, W)


@torch.no_grad()
def _forward(model: nn.Module, x: Tensor, pe_window: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    One forward pass: depth (B, 1, H, W) and class probabilities (B, C, H, W).
    pe_window: frame-consistent position embeddings for a crop of a larger frame (AB1); None = standard.
    """
    x_padded, (top, left, H, W) = pad_to_multiple(x, 14)
    if pe_window is not None:
        pe_window = pe_window.clone()
        pe_window[:, 0] -= top
        pe_window[:, 1] -= left

    with autocast():
        out = model(x_padded) if pe_window is None else model(x_padded, pe_window=pe_window)

    depth = out["depth"][..., top:top + H, left:left + W].float()
    probs = out["seg"][..., top:top + H, left:left + W].float().softmax(1)
    return depth, probs


@torch.no_grad()
def predict(model: nn.Module, img: Tensor, scales: Sequence[float] = (1.0,), flip: bool = False) -> Tuple[Tensor, Tensor]:
    """Predict depth and segmentation probabilities at the input resolution.

    Args:
        model: the model.
        img: normalized input (B, 3, H, W).
        scales: multi-scale test for segmentation (depth is taken at scale 1 only).
        flip: average with the horizontally flipped prediction.
    """
    H, W = img.shape[-2:]
    depth = None
    probs = 0

    for s in scales:
        if s == 1.0:
            x = img
        else:
            x = F.interpolate(img, size=(int(round(H * s)), int(round(W * s))), mode="bilinear", align_corners=False)

        d, p = _forward(model, x)

        if flip:
            d_flip, p_flip = _forward(model, x.flip(-1))
            d = (d + d_flip.flip(-1)) / 2
            p = (p + p_flip.flip(-1)) / 2

        if s == 1.0:
            depth = d
        else:
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)

        probs = probs + p

    if depth is None:
        depth = _forward(model, img)[0]

    return depth, probs / len(scales)


def predict_slide(model: nn.Module, img: Tensor, crop: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    """Sliding-window inference at the training crop size (test-time augmentation).

    Overlapping crops (2 rows x 3 columns: top/bottom, left/centre/right) are predicted one by one and
    averaged where they overlap. Same idea as SwinMTL's --shift_window_test (SwinMTL/trainMulti.py).
    The model then sees the token grid it was trained on, instead of the larger grid of the full frame
    (see notebook 03, section 8.9).

    Args:
        model: the model.
        img: normalized input (1, 3, H, W).
        crop: (h, w) of the training crop.
    """
    H, W = img.shape[-2:]
    ch, cw = min(crop[0], H), min(crop[1], W)
    depth = None
    probs = None
    count = torch.zeros(1, 1, H, W, device=img.device)

    frame_pe = CFG.get("frame_pe", False)
    for y in sorted({0, H - ch}):
        for x in sorted({0, (W - cw) // 2, W - cw}):
            if frame_pe:
                window = torch.tensor([[y, x, H, W]], dtype=torch.float32, device=img.device)
                d, p = _forward(model, img[..., y:y + ch, x:x + cw], window)
            else:
                d, p = predict(model, img[..., y:y + ch, x:x + cw])
            if depth is None:
                depth = torch.zeros(1, 1, H, W, device=img.device)
                probs = torch.zeros(1, p.shape[1], H, W, device=img.device)
            depth[..., y:y + ch, x:x + cw] += d.float()
            probs[..., y:y + ch, x:x + cw] += p.float()
            count[..., y:y + ch, x:x + cw] += 1

    return depth / count, probs / count


def cs_to_eval(gt_seg: Tensor) -> Tensor:
    """
    Cityscapes GT for the standard 19-class metric: void -> ignore.
    """
    gt = gt_seg.clone()
    gt[gt == CFG["datasets"]["cityscapes"]["void_class"]] = 255
    return gt


def eigen_crop_mask(H: int, W: int) -> Tensor:
    """
    Eigen crop [45:471, 41:601] of 480x640, as fractions so it works at any resolution.
    """
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    mask[round(45 / 480 * H):round(471 / 480 * H), round(41 / 640 * W):round(601 / 640 * W)] = True
    return mask


def _eval_loader(dataset: Dataset, limit: Optional[int]) -> DataLoader:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    return DataLoader(Subset(dataset, range(n)), batch_size=1, num_workers=min(4, CFG["num_workers"]))


@torch.no_grad()
def eval_cityscapes(model: nn.Module, limit: Optional[int] = None, ms: bool = False, slide: bool = False) -> dict:
    """Run every Cityscapes protocol in one pass over the val set.

    Args:
        model: the model.
        limit: evaluate only the first images (quick validation).
        ms: also run the multi-scale + flip segmentation test.
        slide: also run every protocol with sliding-window inference at the training crop size (TTA).
    """
    dataset_cfg = CFG["datasets"]["cityscapes"]
    model.eval()
    dataset = make_dataset("cityscapes", dataset_cfg["eval_split"], train=False)

    # metrics init
    depth_512 = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    depth_full = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    seg = SegMeter(dataset_cfg["eval_classes"])
    seg_ms = SegMeter(dataset_cfg["eval_classes"])
    seg_swinmtl = MeanMeter()
    depth_512_slide = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    depth_full_slide = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    seg_slide = SegMeter(dataset_cfg["eval_classes"])
    seg_swinmtl_slide = MeanMeter()

    for batch in tqdm(_eval_loader(dataset, limit), desc="eval cityscapes", leave=False):
        img = batch["image"].to(device)
        gt_depth = batch["depth"][0].to(device)
        gt_seg = batch["seg"][0].to(device)
        H, W = gt_depth.shape

        depth, probs = predict(model, img)
        h, w = depth.shape[-2:]

        # Table I: 512x1024
        gt_depth_small = F.interpolate(gt_depth[None, None], size=(h, w), mode="nearest")[0, 0]
        depth_512.update(depth[0, 0], gt_depth_small)

        # Table II: full resolution
        depth_up = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        depth_full.update(depth_up, gt_depth)

        # segmentation: standard (19 classes, void ignored) and SwinMTL per-image (20 classes)
        probs_full = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[0]
        seg.update(probs_full[:dataset_cfg["eval_classes"]].argmax(0), cs_to_eval(gt_seg))
        seg_swinmtl.update(swinmtl_eval_seg(probs_full.argmax(0), gt_seg, dataset_cfg["num_classes"]))

        # multi-scale + flip segmentation
        if ms:
            probs_ms = predict(model, img, scales=CFG["ms_scales"], flip=True)[1]
            probs_ms = F.interpolate(probs_ms, size=(H, W), mode="bilinear", align_corners=False)[0]
            seg_ms.update(probs_ms[:dataset_cfg["eval_classes"]].argmax(0), cs_to_eval(gt_seg))

        # sliding-window inference at the training crop size (TTA)
        if slide:
            depth_s, probs_s = predict_slide(model, img, dataset_cfg["crop"])
            depth_512_slide.update(depth_s[0, 0], gt_depth_small)
            depth_full_slide.update(F.interpolate(depth_s, size=(H, W), mode="bilinear", align_corners=False)[0, 0], gt_depth)
            probs_s = F.interpolate(probs_s, size=(H, W), mode="bilinear", align_corners=False)[0]
            seg_slide.update(probs_s[:dataset_cfg["eval_classes"]].argmax(0), cs_to_eval(gt_seg))
            seg_swinmtl_slide.update(swinmtl_eval_seg(probs_s.argmax(0), gt_seg, dataset_cfg["num_classes"]))

    results = {
        "cs_depth_512x1024": depth_512.result(),
        "cs_depth_full": depth_full.result(),
        "cs_seg": seg.result(),
        "cs_seg_swinmtl": seg_swinmtl.result()
    }
    if ms:
        results["cs_seg_ms"] = seg_ms.result()
    if slide:
        results["cs_depth_512x1024_slide"] = depth_512_slide.result()
        results["cs_depth_full_slide"] = depth_full_slide.result()
        results["cs_seg_slide"] = seg_slide.result()
        results["cs_seg_swinmtl_slide"] = seg_swinmtl_slide.result()

    return results


@torch.no_grad()
def eval_nyu(model: nn.Module, limit: Optional[int] = None, slide: bool = False) -> dict:
    """Run the NYUv2 protocols: Eigen crop, 10 m, 40-class mIoU.

    Args:
        model: the model.
        limit: evaluate only the first images (quick validation).
        slide: also run the protocols with sliding-window inference at the training crop size (TTA).
    """
    dataset_cfg = CFG["datasets"]["nyu"]
    model.eval()
    dataset = make_dataset("nyu", dataset_cfg["eval_split"], train=False)

    # metrics init
    depth_meter = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    depth_flip_meter = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    seg = SegMeter(dataset_cfg["eval_classes"])
    seg_swinmtl = MeanMeter()
    depth_slide_meter = DepthMeter(dataset_cfg["min_depth"], dataset_cfg["max_depth"])
    seg_slide = SegMeter(dataset_cfg["eval_classes"])
    seg_swinmtl_slide = MeanMeter()

    for batch in tqdm(_eval_loader(dataset, limit), desc="eval nyu", leave=False):
        img = batch["image"].to(device)
        gt_depth = batch["depth"][0].to(device)
        gt_seg = batch["seg"][0].to(device)
        H, W = gt_depth.shape
        mask = eigen_crop_mask(H, W)

        depth, probs = predict(model, img)
        depth_flip = predict(model, img, flip=True)[0]

        # depth, without and with flip averaging
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        depth_flip = F.interpolate(depth_flip, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        depth_meter.update(depth, gt_depth, mask)
        depth_flip_meter.update(depth_flip, gt_depth, mask)

        # segmentation
        pred_seg = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0)
        seg.update(pred_seg, gt_seg)
        seg_swinmtl.update(swinmtl_eval_seg(pred_seg, gt_seg, dataset_cfg["num_classes"]))

        # sliding-window inference at the training crop size (TTA)
        if slide:
            depth_s, probs_s = predict_slide(model, img, dataset_cfg["crop"])
            depth_slide_meter.update(F.interpolate(depth_s, size=(H, W), mode="bilinear", align_corners=False)[0, 0], gt_depth, mask)
            pred_seg_s = F.interpolate(probs_s, size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0)
            seg_slide.update(pred_seg_s, gt_seg)
            seg_swinmtl_slide.update(swinmtl_eval_seg(pred_seg_s, gt_seg, dataset_cfg["num_classes"]))

    results = {
        "nyu_depth": depth_meter.result(),
        "nyu_depth_flip": depth_flip_meter.result(),
        "nyu_seg": seg.result(),
        "nyu_seg_swinmtl": seg_swinmtl.result()
    }
    if slide:
        results["nyu_depth_slide"] = depth_slide_meter.result()
        results["nyu_seg_slide"] = seg_slide.result()
        results["nyu_seg_swinmtl_slide"] = seg_swinmtl_slide.result()
    return results


def quick_val(model: nn.Module, dataset: str, n: int) -> dict:
    """
    Light validation used during training: RMSE, AbsRel and mIoU on the first n images.
    """
    if dataset == "cityscapes":
        results = eval_cityscapes(model, limit=n)
        depth_key, seg_key = "cs_depth_full", "cs_seg"
    else:
        results = eval_nyu(model, limit=n)
        depth_key, seg_key = "nyu_depth", "nyu_seg"

    return dict(
        rmse=results[depth_key]["rmse"],
        abs_rel=results[depth_key]["abs_rel"],
        mIoU=results[seg_key]["mIoU"]
    )


@torch.no_grad()
def efficiency(model: nn.Module, size: Tuple[int, int], n_warmup: int = 10, n_runs: int = 50) -> dict:
    """Parameters, FLOPs (counting SDPA attention) and FPS at a given input size.

    Args:
        model: the model.
        size: input size (h, w).
        n_warmup: forward passes before timing.
        n_runs: timed forward passes (batch 1, bf16).
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, *size, device=device)
    x_padded = pad_to_multiple(x)[0]

    # FLOPs
    with FlopCounterMode(display=False) as flop_counter:
        with autocast():
            model(x_padded)
    flops = flop_counter.get_total_flops()

    # FPS
    for _ in range(n_warmup):
        _forward(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(n_runs):
        _forward(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    stats = torchinfo.summary(model, verbose=0)   # parameter counts only (no input: no forward pass)
    return dict(
        params_M=stats.total_params / 1e6,
        trainable_M=stats.trainable_params / 1e6,
        GFLOPs=flops / 1e9,
        GMACs=flops / 2e9,
        fps=n_runs / elapsed,
        input=f"{size[0]}x{size[1]}",
        device=str(device)
    )


configure()
