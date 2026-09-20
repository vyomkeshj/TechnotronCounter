"""Image normalisation, letterboxing and the augmentation set both pipelines use."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)   # REQUIRED in anything a DataLoader/ProcessPool worker imports

MEAN = np.array([0.45, 0.45, 0.45], np.float32)
STD = np.array([0.23, 0.23, 0.23], np.float32)


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)))


def to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip((x * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, size: int, pad_value: int = 0, interp: int = cv2.INTER_LINEAR):
    """Aspect-preserving resize + pad to size x size. Returns (img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size) + img.shape[2:], pad_value, img.dtype)
    out[py:py + nh, px:px + nw] = r
    return out, s, px, py


def pad_to_multiple(img: np.ndarray, m: int = 16) -> tuple[np.ndarray, int, int]:
    """Pad bottom/right with zeros so both sides divide by m (fully-convolutional inference)."""
    h, w = img.shape[:2]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return img, h, w


def photometric(img: np.ndarray) -> np.ndarray:
    """Brightness/contrast/gamma/colour cast. STRONG photometric aug was essential:
    light aug overfits the small labelled sets."""
    f = img.astype(np.float32)
    if random.random() < 0.8:
        f = f * (1.0 + random.uniform(-0.4, 0.4)) + random.uniform(-45, 45)
    if random.random() < 0.5:
        f = 255.0 * np.clip(f / 255.0, 0, 1) ** random.uniform(0.5, 1.8)
    if random.random() < 0.4:
        f = f * np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray) -> np.ndarray:
    """Blur / sensor noise / JPEG."""
    if min(img.shape[:2]) < 16:
        return img
    if random.random() < 0.4:
        k = random.choice([3, 5, 7])
        try:
            img = cv2.GaussianBlur(img, (k, k), 0)
        except cv2.error:
            pass
    if random.random() < 0.3:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(4, 14), img.shape),
                      0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 75)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


_PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
                     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0]], np.uint8)


def colorize(lab: np.ndarray) -> np.ndarray:
    out = np.zeros(lab.shape + (3,), np.uint8)
    m = lab > 0
    out[m] = _PALETTE[(lab[m] - 1) % len(_PALETTE)]
    return out
