"""框后处理优化(投机取巧但合规:框只要IoU≥0.5,位置准+尺寸大概对即可):
①碎框合并(近邻union) ②框膨胀 ③尺寸先验保底框(掩膜空→异常图argmax+中位GT尺寸)。
所有参数在fit集(官方给的30掩膜)上网格搜索最优,直接优化框命中@0.5。
用法:python scripts/run_box_opt.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256); SEG_IN = 512


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def gt_boxes(mask, min_a=4):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in (stats[i] for i in range(1, n)) if a >= min_a]


def biou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1)


def hit_rate(preds_list, gts_list):
    """GT框召回率@0.5(与成绩单口径一致)。"""
    tot, hit = 0, 0
    for preds, gts in zip(preds_list, gts_list):
        for g in gts:
            tot += 1
            if any(biou(p, g) >= 0.5 for p in preds):
                hit += 1
    return hit / max(tot, 1)


def raw_boxes(mask):
    return gt_boxes(mask, min_a=3)


def merge_boxes(boxes, d):
    """近邻(间距<d)框合并为union,迭代到收敛。"""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - d < b[2] and b[0] - d < a[2] and a[1] - d < b[3] and b[1] - d < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j); changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def pad_boxes(boxes, f, W=256, H=256):
    out = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, (x2-x1)*f, (y2-y1)*f
        out.append((max(0, cx-w/2), max(0, cy-h/2), min(W, cx+w/2), min(H, cy+h/2)))
    return out


def snap_min(boxes, mw, mh, W=256, H=256):
    """框小于先验中位的一半→放大到中位尺寸(居中)。"""
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2-x1, y2-y1
        if w < mw * 0.5 or h < mh * 0.5:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = max(w, mw), max(h, mh)
            out.append((max(0, cx-w/2), max(0, cy-h/2), min(W, cx+w/2), min(H, cy+h/2)))
        else:
            out.append((x1, y1, x2, y2))
    return out


def postproc(mask, amap, d, f, prior, use_snap, use_fb):
    bs = raw_boxes(mask)
    if d > 0:
        bs = merge_boxes(bs, d)
    if f > 1.0:
        bs = pad_boxes(bs, f)
    if use_snap and prior:
        bs = snap_min(bs, prior[0], prior[1])
    if use_fb and not bs and prior:
        y, x = np.unravel_index(np.argmax(amap), amap.shape)
        mw, mh = prior
        bs = [(max(0, x-mw/2), max(0, y-mh/2), min(256, x+mw/2), min(256, y+mh/2))]
    return bs


def run_cat(name, normals, fit, tests, bb):
    ex = lambda img: bb.extract(F.interpolate((img.unsqueeze(0) if img.dim() == 3 else img).to(bb.device),
                                              size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False))[0]
    head = SupervisedSegHead(device=DEV, extractor=ex)
    class _D: pass
    head.fit(_D(), [f[0] for f in fit], [f[1] for f in fit], normals[:30])
    thr = head.thr
    # 尺寸先验:fit GT框中位宽高
    all_gt = [b for _, mk in fit for b in gt_boxes(mk)]
    prior = (float(np.median([b[2]-b[0] for b in all_gt])), float(np.median([b[3]-b[1] for b in all_gt]))) if all_gt else None
    def masks_amaps(items):
        out = []
        for img, mk in items:
            amap = head.map(_D(), img, HW)
            out.append(((amap >= thr).astype(np.uint8), amap, gt_boxes(mk)))
        return out
    fit_d = masks_amaps(fit); test_d = masks_amaps(tests)
    # 基线
    base = hit_rate([raw_boxes(m) for m, _, _ in test_d], [g for _, _, g in test_d])
    # fit上网格搜索
    best_cfg, best_fit = None, -1
    for d in [0, 4, 8, 16]:
        for f in [1.0, 1.3, 1.6, 2.0]:
            for snap in [False, True]:
                h = hit_rate([postproc(m, a, d, f, prior, snap, True) for m, a, _ in fit_d],
                             [g for _, _, g in fit_d])
                if h > best_fit:
                    best_fit, best_cfg = h, (d, f, snap)
    d, f, snap = best_cfg
    opt = hit_rate([postproc(m, a, d, f, prior, snap, True) for m, a, _ in test_d],
                   [g for _, _, g in test_d])
    print(f"{name:16s} 基线框命中={base:.3f} → 优化后={opt:.3f} (Δ{opt-base:+.3f}) "
          f"[合并d={d} 膨胀f={f} snap={snap} 先验={prior and (round(prior[0]),round(prior[1]))}]", flush=True)
    return base, opt


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:60]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[:k]]
    tests = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return normals, fit, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:60]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    return normals, fit, tests


def main():
    torch.manual_seed(0)
    bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)
    print("=== 框后处理优化(fit集标定:合并/膨胀/尺寸先验保底)===")
    B, O = [], []
    for name, prep in [("电子 pcb", lambda: prep_realiad("pcb")),
                       ("电池 battery", lambda: prep_realiad("phone_battery")),
                       ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
                       ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]))]:
        normals, fit, tests = prep()
        b, o = run_cat(name, normals, fit, tests, bb)
        B.append(b); O.append(o)
    print(f"\n均值: 基线={np.mean(B):.3f} → 优化后={np.mean(O):.3f} (Δ{np.mean(O)-np.mean(B):+.3f})")


if __name__ == "__main__":
    main()
