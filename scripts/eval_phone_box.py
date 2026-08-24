"""真实手机屏数据(data/phone)上的**框命中@0.5**——用户口径:"有框也行,能指出位置就行"。

**这个数的前提必须一起报,否则会误导**:赛题协议给100张正常图,而本数据集全库只有
**7张**正常图(还是Roboflow增广出来的重复)。EAD学生、DINO门、像素阈值全都在正常图
分布上标定,7张是严重饥饿状态。所以这里测的是**下界**:"正常样本被饿到极限时,
定位还剩多少能力"。真实赛场有100张正常图,只会比这个好。

同理**图级准确率在本数据集上无法测**:7张正常图全部拿去fit了,测试侧一张正常图都
不剩,误报率无从谈起。这里只报两个能诚实测出来的:
  ①检出率(缺陷图里有多少被判为缺陷)  ②框命中@0.5(判出来的框有没有指对位置)

用法:PYTHONPATH=. python scripts/eval_phone_box.py [测试张数]
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
from global_context.eval_global_branch import gt_boxes, box_hit, _box_iou as _iou

ROOT = "data/phone"


def _load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _boxes_mask(shape_hw, lab_path):
    H, W = shape_hw
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


def _pairs(split, want, skip=0):
    """返回 [(img, box_mask)],跳过空标注(那是正常图)。"""
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{split}/images/*"))[skip:]:
        img = _load(f)
        mk = _boxes_mask(img.shape[-2:], f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        if not mk.any():
            continue
        out.append((img, mk))
        if len(out) >= want:
            break
    return out


def _normals():
    out = []
    for split in ("train", "valid", "test"):
        for f in sorted(glob.glob(f"{ROOT}/{split}/labels/*.txt")):
            if os.path.getsize(f) == 0:
                out.append(_load(f.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"))
    return out


def main(n_test=100):
    torch.manual_seed(0)
    normals = _normals()
    fit = _pairs("train", 30)
    test = _pairs("valid", n_test)
    print(f"正常图 {len(normals)} 张(赛题协议是100张——严重饥饿,本结果是下界) / "
          f"fit缺陷 {len(fit)} / 测试缺陷 {len(test)}", flush=True)

    t0 = time.time()
    det = CompetitionLargeDetector(compile_infer=True)
    det.fit_fewshot(normals, [i for i, _ in fit], defect_masks=[m for _, m in fit])
    print(f"fit完成 {time.time()-t0:.0f}s  type_head={'就绪' if det.type_head else '未启用'}", flush=True)

    for im, _ in test[:5]:
        det.locate(im)                                     # 预热,不计时

    hits, lat, n_det = [], [], 0
    # 诊断:命中率为0时要能区分两种失效——"框位置对但太粗/太大"(bestIoU有值但<0.5)
    # 还是"框根本指错地方"(bestIoU≈0)。再记预测掩膜占图比例,饿正常图会让模型把
    # 整片区域都标成异常,那种情况下IoU分母被撑爆、必然全灭。
    best_ious, pred_frac, n_boxes = [], [], []
    for img, gt in test:
        t1 = time.time()
        o = det.locate(img)
        lat.append((time.time() - t1) * 1000)
        n_det += bool(o["is_defect"])
        if o.get("mask") is None:
            hits.append(0.0); continue
        mk = o["mask"]
        gt_r = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                              size=mk.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        gbs = gt_boxes(gt_r)
        h = box_hit(o["boxes"], gbs)
        hits.append(h if h is not None else 0.0)
        pred_frac.append(float(mk.astype(bool).mean()))
        n_boxes.append(len(o["boxes"]))
        for g in gbs:
            best_ious.append(max([_iou(pb[:4], g) for pb in o["boxes"]], default=0.0))
    n = len(test)
    print(f"\n=== 手机屏真实数据(正常图仅{len(normals)}张,下界) ===", flush=True)
    print(f"检出率(缺陷图判为缺陷)  {n_det}/{n} = {n_det/max(n,1):.1%}", flush=True)
    print(f"框命中@0.5              {np.mean(hits):.3f}", flush=True)
    print(f"延时  中位={np.median(lat):.0f}ms  p90={np.percentile(lat,90):.0f}ms", flush=True)
    if best_ious:
        b = np.array(best_ious)
        print(f"诊断 每个GT框的最佳IoU:  中位={np.median(b):.3f}  p90={np.percentile(b,90):.3f}  "
              f"最大={b.max():.3f}  |  完全无重叠(IoU=0)的占 {(b==0).mean():.0%}", flush=True)
        print(f"诊断 预测掩膜占图比例:  中位={np.median(pred_frac):.1%}  最大={max(pred_frac):.1%}  "
              f"(GT框中位只占2.7%)  |  每图预测框数 中位={np.median(n_boxes):.0f}", flush=True)
    print("注:图级准确率在本数据集不可测——7张正常图全部用于fit,测试侧无正常样本,误报率无从谈起。",
          flush=True)
    print("PHONE BOX OK", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
