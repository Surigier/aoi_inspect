"""真实手机屏数据上的类型归属验证(data/phone,Roboflow YOLO格式)。

**这个数据集只能用来验类型归属,不能用来报检测/定位成绩**:
  ①全库只有7张正常图(还是增广出来的重复),而fit协议要100张正常图;
  ②标注是检测框不是掩膜,拿它算IoU会被系统性压低。

也不预设"oil该映射到赛题哪一类"——那正是metal_nut踩过的坑(基准标签自己就有噪声,
在上面调提示词等于过拟合)。这里只看**VLM把数据集的3个类各自映到赛题5类的哪一类、
以及映得一不一致**,让映射自己浮出来,再交给人判断合不合理。

用法:PYTHONPATH=. python scripts/eval_phone_type.py [每类张数]
"""
import collections
import glob
import os
import sys
import numpy as np
import torch
from PIL import Image
from aoi.vlm_type import label_defect_types

ROOT = "data/phone/train"
PREFIX = {"Oil": "油污", "Scr": "划痕", "Sta": "污渍"}


def _load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _boxes_mask(img, lab_path):
    """YOLO框 → 矩形掩膜。多个框合并成一张掩膜(和locate输出的多连通域掩膜同形态)。"""
    _, H, W = img.shape
    m = np.zeros((H, W), np.uint8)
    if not os.path.exists(lab_path):
        return m
    for line in open(lab_path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = [float(x) for x in p[1:5]]
        x0, x1 = int((cx - w / 2) * W), int((cx + w / 2) * W)
        y0, y1 = int((cy - h / 2) * H), int((cy + h / 2) * H)
        m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 1
    return m


def main(n_per=20):
    rows = {}
    for pre, cn in PREFIX.items():
        imgs, msks = [], []
        for f in sorted(glob.glob(f"{ROOT}/images/{pre}_*"))[:n_per * 3]:
            img = _load(f)
            mk = _boxes_mask(img, f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
            if not mk.any():
                continue                                   # 空标注=正常图,跳过
            imgs.append(img); msks.append(mk)
            if len(imgs) >= n_per:
                break
        labs = label_defect_types(imgs, msks, verbose=False)
        c = collections.Counter(l for l in labs if l) if labs else collections.Counter()
        rows[cn] = c
        top = c.most_common(1)
        consist = f"{top[0][1]}/{sum(c.values())}" if top else "0/0"
        print(f"{cn}({pre}, n={len(imgs)}): 最一致映射={top[0][0] if top else '—'} {consist}  全部={dict(c)}",
              flush=True)
    print("\n=== 映射矩阵(数据集类 → 赛题5类) ===", flush=True)
    for cn, c in rows.items():
        tot = max(sum(c.values()), 1)
        print(f"  {cn:4s} " + "  ".join(f"{k}:{v/tot:.0%}" for k, v in c.most_common()), flush=True)
    print("PHONE TYPE OK", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
