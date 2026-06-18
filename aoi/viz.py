import numpy as np
from PIL import Image


def overlay_heatmap(image_chw, amap, alpha: float = 0.5) -> Image.Image:
    """把异常图以红色半透明叠到原图上。
    image_chw: (3,H,W) [0,1] 的 tensor 或 array;amap: (h,w) 异常图。返回 RGB PIL.Image。"""
    img = np.asarray(image_chw, dtype=np.float32).transpose(1, 2, 0)   # H,W,3
    h, w = img.shape[:2]
    a = np.asarray(amap, dtype=np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)                     # 归一化到 [0,1]
    a_resized = np.asarray(Image.fromarray((a * 255).astype("uint8")).resize((w, h))) / 255.0
    heat = np.zeros_like(img)
    heat[..., 0] = a_resized                                           # 红色通道
    out = (1.0 - alpha) * img + alpha * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"))
