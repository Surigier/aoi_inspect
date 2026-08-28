"""手机屏专用快速台架 —— 小规模、快迭代,用来筛方法。

为什么单独做一个:手机屏(MSD)是赛题的隐藏域本身,最该用它筛方法。
规模刻意做小(fit 15正常+30缺陷,test默认200张),一轮比全量快很多,便于横向比多个配置。

数据与限制(必须写在脸上):
  正常图:MSD官方good.zip **只有20张**(1920×1080)→ 15张fit / 5张test。
          **误报率只能弱估计**(5张正常,一张翻转=20%),看趋势不看绝对值。
  缺陷图:phone_best 1200张(640×360),GT是**检测框**转的矩形掩膜 →
          **IoU被系统性压低**(我们预测贴合缺陷的掩膜,GT是外接矩形),
          **以框命中@0.5为主指标**,IoU仅供横向比较。
  尺寸:good缩到640×360与缺陷图一致(16:9,不变形)。

用法:
  PYTHONPATH=. python scripts/run_exam_phone.py [测试缺陷数=200] [--seg-gate] [--dino-seg] [--per-mode]
"""
import glob
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import gt_boxes, box_hit

W, H = 640, 360
HW = (256, 256)
N_FIT_NORM, N_FIT_DEF, N_TEST_NORM = 15, 30, 5
# 数据结构的硬事实(看图确认过,不是推测):MSD的20张"正常图"是**20个不同手机型号**,
# 一个型号一张——白边框/黑边框/不同尺寸/不同摄像头位置全不一样。
# 后果:我们"用100张正常图给**这一个产品**建模"的前提在这份数据上不成立,
# 拿没见过的机型去测必然全判异常。**图级误报率在这份数据上测不了**,不要报。
# 但"给定一张缺陷图,能不能找到并框准缺陷"**不依赖正常建模**,这才是这份数据的用武之地,
# 而且phone_best有1200张缺陷图,足够把这个能力测扎实。


def _img(p):
    im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)


def _mask(lab):
    m = np.zeros(HW, np.uint8)
    if not os.path.exists(lab):
        return m
    h, w = HW
    for line in open(lab):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = [float(x) for x in p[1:5]]
        m[max(0, int((cy - bh / 2) * h)):min(h, int((cy + bh / 2) * h) + 1),
          max(0, int((cx - bw / 2) * w)):min(w, int((cx + bw / 2) * w) + 1)] = 1
    return m


def main(n_test_def=1000, **kw):
    SEED = int(os.environ.get("EXAM_SEED", "0"))
    torch.manual_seed(SEED)
    goods = sorted(glob.glob("data/msd_good/good/*.png"))
    defs = []
    for s in ("train", "val", "test"):
        for f in sorted(glob.glob(f"data/phone_best/{s}/images/*")):
            lab = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if os.path.exists(lab) and os.path.getsize(lab) > 0:
                defs.append((f, lab))
    random.Random(SEED + 1).shuffle(defs)                     # 固定种子打乱:避免按目录顺序造成类型偏置
    print(f"正常{len(goods)}张(→{N_FIT_NORM}fit/{N_TEST_NORM}test) 缺陷{len(defs)}张", flush=True)

    fit_n = [_img(p) for p in goods[:N_FIT_NORM]]
    test_n = [_img(p) for p in goods[N_FIT_NORM:N_FIT_NORM + N_TEST_NORM]]
    fit_d = [(_img(f), _mask(l)) for f, l in defs[:N_FIT_DEF]]
    test_d = defs[N_FIT_DEF:N_FIT_DEF + n_test_def]    # **fit没见过的**

    det = CompetitionLargeDetector(**kw)
    print(f"配置: {kw or '基线'}", flush=True)
    t0 = time.time()
    det.fit_fewshot(fit_n, [i for i, _ in fit_d], defect_masks=[m for _, m in fit_d])
    print(f"fit完成 {time.time()-t0:.0f}s 阈值={det.decision_threshold():.4f}", flush=True)
    print(f"!! 延时自适应裁剪={getattr(det,'lat_trimmed',None)} 探针={getattr(det,'lat_probe_ms',None)} "
          f"DINO门={'在' if getattr(det,'_dino',None) is not None else '**已被砍**'} "
          f"SAM={'在' if det.sam is not None else '已砍'}", flush=True)

    # 测试流:**全部fit没见过的缺陷图** + 5张留出正常图(仅作弱参考,不并入主指标)
    stream = [(d, True) for d in test_d] + [((None, None), False)] * N_TEST_NORM
    random.Random(SEED + 2).shuffle(stream)
    tp = fn = fp = tn = 0; hits = []; ious = []; lats = []; sc = []; lb = []
    ni = 0
    for item, is_def in stream:
        if is_def:
            img = _img(item[0]); gt = _mask(item[1])
        else:
            img = test_n[ni % len(test_n)]; gt = None; ni += 1
        t1 = time.time(); o = det.locate(img); lats.append((time.time() - t1) * 1000)
        # 扫描必须用**与判决同口径**的分:o["score"]是原始EAD分,而判决走的是
        # DINO融合后的z分,两者尺度差几个数量级(实测"当前阈值23140 vs 最优2.2"就是
        # 拿EAD原始分比融合z阈值比出来的,那几行数字作废)。frame_score()与
        # decision_threshold()是配对的同口径接口。
        sc.append(float(det.frame_score(img))); lb.append(is_def)
        pred = bool(o["is_defect"])
        if is_def and pred: tp += 1
        elif is_def: fn += 1
        elif pred: fp += 1
        else: tn += 1
        if is_def:
            mk = o.get("mask")
            if mk is None:
                hits.append(0.0); ious.append(0.0)
            else:
                g = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                                   size=mk.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
                p_ = mk.astype(bool); g_ = g.astype(bool)
                TP = int((p_ & g_).sum()); FP2 = int((p_ & ~g_).sum()); FN2 = int((~p_ & g_).sum())
                ious.append(TP / max(TP + FP2 + FN2, 1))
                h = box_hit(o["boxes"], gt_boxes(g)); hits.append(h if h is not None else 0.0)
    nd = tp + fn
    hh = np.array(hits); ii = np.array(ious)
    print(f"\n=== 手机屏(隐藏域)缺陷发现能力 · 测试缺陷 {nd} 张 ===", flush=True)
    print(f"【主指标】检出率(召回) = {tp}/{nd} = {tp/max(nd,1):.1%}", flush=True)
    print(f"【主指标】框命中@0.5   = {hh.mean():.3f}   (GT是检测框,这是最该看的定位指标)", flush=True)
    print(f"          含漏检IoU    = {ii.mean():.3f}   (GT是矩形框→系统性偏低,仅供横向比较)", flush=True)
    print(f"          检出的那些框命中 = {hh[hh>0].mean() if (hh>0).any() else 0:.3f}"
          f"(剔除漏检,看纯定位质量)", flush=True)
    print(f"延时 中位={np.median(lats):.0f}ms p90={np.percentile(lats,90):.0f}ms", flush=True)
    print(f"\n[弱参考] 留出正常图 {fp+tn} 张 → 误判 {fp} 张。", flush=True)
    print(f"  **不作为指标**:这20张正常图是20个不同手机型号(已看图确认),", flush=True)
    print(f"  没有一致的『正常』类可建模,拿没见过的机型测必然报异常——是数据结构问题,不是方法问题。", flush=True)
    print(f"RESULT seed={SEED} recall={tp/max(nd,1):.4f} hit={hh.mean():.4f} "
          f"iou={ii.mean():.4f} hit_det={hh[hh>0].mean() if (hh>0).any() else 0:.4f} "
          f"lat_med={np.median(lats):.1f}", flush=True)
    print("PHONE_EXAM OK", flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    kw = {}
    if "--seg-gate" in sys.argv: kw["seg_gate"] = True
    if "--dino-seg" in sys.argv: kw["dino_seg"] = True
    if "--per-mode" in sys.argv: kw["per_mode_gate"] = True
    main(int(a[0]) if a else 1000, **kw)
