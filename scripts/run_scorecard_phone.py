"""真手机屏成绩单 —— 隐藏测试域最直接的证据。

数据配对(必须这样配,不能乱配):
  正常图 = MSD官方 good.zip 的20张(1920×1080原始分辨率)
  缺陷图 = data/phone_best(960张,640×360,**16:9长宽比与good一致**)
  ✗ 不用 data/phone:它被Roboflow拉成640×640正方形,长宽比破坏,和正常图对不上,
    特征分布会系统性错位。

**两条必须写在脸上的限制**:
1. **只有20张正常图,协议要100张**——是协议的1/5。所以本表是**下界**,不是能力上限。
2. **测试侧没有正常样本**(20张全部用于fit)→ **图级准确率/误报率不可测**,本表不报。
   报的是:检出率(召回)、含漏检IoU、框命中@0.5、延时。
3. GT是**检测框**不是像素掩膜 → IoU被系统性压低(我们预测的是贴合缺陷的掩膜,
   GT是外接矩形)。**看这份表应以框命中@0.5为主**,IoU仅供参考。

用法:PYTHONPATH=. python scripts/run_scorecard_phone.py [测试张数=100]
"""
import glob
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import gt_boxes, box_hit

GOOD = "data/msd_good/good"
DEF_ROOT = "data/phone_best"
W, H = 640, 360                      # 与 phone_best 一致;good 的1920×1080按同长宽比缩到这里
HW = (256, 256)                      # 评测掩膜分辨率,与 run_scorecard 同口径


def _load(path, w=W, h=H):
    im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _boxes_mask(lab_path, hw=HW):
    """YOLO框 → 评测分辨率上的矩形掩膜。多框合并。"""
    h, w = hw
    m = np.zeros((h, w), np.uint8)
    if not os.path.exists(lab_path):
        return m
    for line in open(lab_path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = [float(x) for x in p[1:5]]
        x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
        y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
        m[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = 1
    return m


def _defects(split, want, skip=0):
    out = []
    for f in sorted(glob.glob(f"{DEF_ROOT}/{split}/images/*")):
        lab = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lab) and os.path.getsize(lab) > 0:
            out.append((f, lab))
    return out[skip:skip + want]


def main(n_test=100):
    torch.manual_seed(0)
    gf = sorted(glob.glob(f"{GOOD}/*.png")) + sorted(glob.glob(f"{GOOD}/*.jpg"))
    normals = [_load(f) for f in gf]
    fit_pairs = _defects("train", 30)
    test_pairs = _defects("val", n_test) + _defects("test", max(0, n_test - 120))
    test_pairs = test_pairs[:n_test]
    print(f"正常图 {len(normals)}张(协议要100张 → 本表是下界) / fit缺陷 {len(fit_pairs)} / "
          f"测试缺陷 {len(test_pairs)}", flush=True)

    fit_i = [_load(f) for f, _ in fit_pairs]
    fit_m = [_boxes_mask(l) for _, l in fit_pairs]

    t0 = time.time()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"fit完成 {time.time()-t0:.0f}s  type_head={'就绪' if det.type_head else '未启用'}", flush=True)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    for f, _ in test_pairs[:3]:
        det.locate(_load(f))                                   # 预热,不计时

    hits, ious, lat, n_det = [], [], [], 0
    types = {}
    for f, l in test_pairs:
        img = _load(f); gt = _boxes_mask(l)
        t1 = time.time(); o = det.locate(img); lat.append((time.time() - t1) * 1000)
        n_det += bool(o["is_defect"])
        if o["is_defect"]:
            types[o["defect_type"]] = types.get(o["defect_type"], 0) + 1
        if o.get("mask") is None:
            hits.append(0.0); ious.append(0.0); continue
        mk = o["mask"]
        g = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                           size=mk.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        p = mk.astype(bool); gb = g.astype(bool)
        TP = int((p & gb).sum()); FP = int((p & ~gb).sum()); FN = int((~p & gb).sum())
        ious.append(TP / max(TP + FP + FN, 1))
        h = box_hit(o["boxes"], gt_boxes(g))
        hits.append(h if h is not None else 0.0)

    n = max(len(test_pairs), 1)
    print(f"\n=== 真手机屏成绩单(MSD good 20张正常 + phone_best 缺陷)===", flush=True)
    print(f"检出率(召回)   {n_det}/{n} = {n_det/n:.1%}", flush=True)
    print(f"框命中@0.5     {np.mean(hits):.3f}   ← 主指标(GT是框)", flush=True)
    print(f"含漏检IoU      {np.mean(ious):.3f}   ← 仅供参考(GT是矩形框,系统性压低)", flush=True)
    print(f"延时           中位={np.median(lat):.0f}ms  p90={np.percentile(lat,90):.0f}ms", flush=True)
    print(f"类型分布       {types}", flush=True)
    print("注:图级准确率不可测——20张正常图全部用于fit,测试侧无正常样本,误报率无从谈起。", flush=True)
    print("PHONE_SCORECARD OK", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
