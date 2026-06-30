"""快速图像加载(按赛题口径:单图加载+预处理计入推理耗时)。
2500² PNG 全解码~167ms 吃光预算;JPEG 用 cv2 半/四分之一分辨率解码(IMREAD_REDUCED)
可降到 14-36ms。反正下游要缩到 ~max_size,提前在解码阶段降分辨率既快又等价。
"""
import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    import pyspng
    _HAS_SPNG = True
except Exception:
    _HAS_SPNG = False
from PIL import Image


def load_fast(path, max_size=1152):
    """读盘 → (3,H,W) float[0,1] tensor,长边≤~max_size。
    PNG:pyspng(~2×PIL,73ms@2500²);JPEG:cv2 reduced 解码(1/2、1/4)少解码;均缩到max_size。"""
    path = str(path)
    if _HAS_SPNG and path.lower().endswith(".png"):
        im = _spng_load(path, max_size)
        if im is not None:
            return _to_tensor(im)
    if _HAS_CV2:
        im = _cv2_load(path, max_size)
        if im is not None:
            return _to_tensor(im)
    # 回退 PIL(JPEG 用 draft 近似降分辨率)
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (max_size, max_size))
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return _to_tensor(np.asarray(img))


def _spng_load(path, max_size):
    """pyspng 快速 PNG 解码(无reduced,全解码后缩放)。"""
    try:
        with open(path, "rb") as f:
            im = pyspng.load(f.read())
    except Exception:
        return None
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    H, W = im.shape[:2]
    if max(H, W) > max_size and _HAS_CV2:
        s = max_size / max(H, W)
        im = cv2.resize(im, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im)


def _cv2_load(path, max_size):
    # 先读 header 拿尺寸,决定 reduced 倍率(仅对 JPEG 真正省解码;PNG 退化为全解码)
    flag = cv2.IMREAD_COLOR
    try:
        h0, w0 = _peek_size(path)
        if h0 and max(h0, w0) >= 4 * max_size:
            flag = cv2.IMREAD_REDUCED_COLOR_4
        elif h0 and max(h0, w0) >= 2 * max_size:
            flag = cv2.IMREAD_REDUCED_COLOR_2
    except Exception:
        pass
    im = cv2.imread(path, flag)
    if im is None:
        return None
    im = im[:, :, ::-1]                                    # BGR→RGB
    H, W = im.shape[:2]
    if max(H, W) > max_size:
        s = max_size / max(H, W)
        im = cv2.resize(im, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im)


def _peek_size(path):
    with Image.open(path) as im:
        w, h = im.size
    return h, w


def _to_tensor(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
