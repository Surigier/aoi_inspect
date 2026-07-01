"""反事实缺陷合成(正常→缺陷):对正常图随机区域施加 CutPaste/色彩/噪声,生成缺陷+掩膜。
扩张监督分割头的训练分布→学通用缺陷特征→跨域泛化(实测跨域 +0.065)。赛题点名合成缺陷。
轻量(纯张量操作),在不计时的 fit 阶段跑。"""
import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def synth_defect(img, rng):
    """正常图 (3,H,W)[0,1] → (缺陷图, mask(H,W){0,1})。
    随机 1-3 个椭圆 blob,各随机施加:CutPaste(错位内容)/色彩偏移/噪声——
    覆盖外观/结构/色彩类缺陷的视觉代理。rng=random.Random 实例。"""
    C, H, W = img.shape
    out = img.clone()
    mask = np.zeros((H, W), np.uint8)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.randint(0, W - 1), rng.randint(0, H - 1)
        ax = rng.randint(max(4, W // 20), max(6, W // 6))
        ay = rng.randint(max(4, H // 20), max(6, H // 6))
        m = _ellipse(H, W, cx, cy, ax, ay, rng.randint(0, 180))
        mb = torch.from_numpy(m > 0).to(img.device)
        mode = rng.choice(["cutpaste", "color", "noise"])
        if mode == "cutpaste":
            dx, dy = rng.randint(-W // 3, W // 3), rng.randint(-H // 3, H // 3)
            src = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
            out[:, mb] = src[:, mb]
        elif mode == "color":
            f = torch.tensor([rng.uniform(0.4, 1.8) for _ in range(3)], device=img.device).view(3, 1)
            out[:, mb] = (out[:, mb] * f).clamp(0, 1)
        else:
            out[:, mb] = (out[:, mb] + torch.randn_like(out[:, mb]) * 0.3).clamp(0, 1)
        mask |= m
    return out, mask


def _ellipse(H, W, cx, cy, ax, ay, ang):
    if _HAS_CV2:
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (cx, cy), (ax, ay), ang, 0, 360, 1, -1)
        return m
    # 回退:轴对齐椭圆(无 cv2)
    yy, xx = np.ogrid[:H, :W]
    return (((xx - cx) / max(1, ax)) ** 2 + ((yy - cy) / max(1, ay)) ** 2 <= 1).astype(np.uint8)
