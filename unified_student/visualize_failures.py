"""诊断性可视化(不是新机制验证,是肉眼看几张真实失败案例):挑breakfast_box(今天
好几次实验里表现最稳定的"0.000无变化"类目)的logical_anomalies测试图,真实跑一遍
生产CompetitionLargeDetector,把原图/真实标注掩膜/模型预测掩膜并排存成图,供肉眼
检查具体错在哪(漏检?定位偏了?还是掩膜本身就没对上缺陷区域?)。

用法:PYTHONPATH=. python unified_student/visualize_failures.py

【肉眼检查结论,2026-07-27】6张测试图(019/009/042/034/058/010.png),4中2漏
(009/010漏检)。关键观察:
1. 6张图的GT缺陷本质都是同一种——右侧格子granola被坚果替换/欠量,漏检的
   009(GT前景246346px)和命中的019(242772px)标注面积几乎相等——不是"缺陷小
   所以漏检"。
2. raw EAD分数:漏检两张(0.633/0.635)非常接近命中中最弱的058(0.654),命中中
   最强的034(1.194)是唯一"整格子全空"的重度案例。分数呈连续谱,漏检的两张
   卡在决策阈值边缘,不是分数骤降或落到完全不相关的区间。
3. 019的预测掩膜(0_pred_overlay.png)大体落在真实坚果混入区域(和GT红色区域
   重叠),但也有不少水果纹理反光造成的小面积散点FP。
**结论**:这次是"信号强度不够、卡在阈值边缘"的margin问题,不是掩膜定位错地方
或存在结构性盲区的bug。和当天另外8条机制路线全部判负的结果互相印证:现有
EAD+DINO局部patch比对范式,对"轻度构成性替换/欠量"这类逻辑异常,原始信号
本来就弱,不是缺了某个具体信号源能补的——这是定性肉眼观察(n=2漏检样本),
不构成可直接改动生产的证据,仅作诊断记录。
"""
import glob
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast

OUT = Path("/tmp/claude-1000/-home-srj-yolo/d8e53301-3b28-4836-8a41-ee22b204462e/scratchpad/failure_vis")
OUT.mkdir(parents=True, exist_ok=True)


def _union_mask(gt_dir):
    m = None
    for mp in sorted(gt_dir.glob("*.png")):
        arr = (np.array(Image.open(mp).convert("L")) > 0)
        m = arr if m is None else (m | arr)
    return m.astype(np.uint8) if m is not None else None


def main():
    torch.manual_seed(0)
    cat = "breakfast_box"
    root = Path(f"data/_dl/mvtec_loco/{cat}")
    normals = [load_fast(p) for p in sorted((root / "train/good").glob("*.png"))[:100]]
    imgs = sorted((root / "test/logical_anomalies").glob("*.png"))
    random.Random(0).shuffle(imgs)
    fit_p, test_p = imgs[:15], imgs[15:]
    fit_i = [load_fast(p) for p in fit_p]
    fit_m = [_union_mask(root / "ground_truth/logical_anomalies" / p.stem) for p in fit_p]

    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"seg_head训成功={det.seg_head.head is not None} thr={det.seg_head.thr}", flush=True)

    for i, p in enumerate(test_p[:6]):
        img = load_fast(p)
        gt = _union_mask(root / "ground_truth/logical_anomalies" / p.stem)
        o = det.locate(img)
        print(f"[{i}] {p.name}: is_defect={o['is_defect']} score={o['score']:.3f} thr={det.decision_threshold():.3f} "
              f"boxes={len(o.get('boxes') or [])} GT前景像素={int(gt.sum())}/{gt.size}", flush=True)

        native = img.permute(1, 2, 0).cpu().numpy()
        native_u8 = (native * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(native_u8).save(OUT / f"{i}_orig.png")

        gt_rs = np.array(Image.fromarray(gt * 255).resize((native_u8.shape[1], native_u8.shape[0]), Image.NEAREST))
        overlay_gt = native_u8.copy()
        overlay_gt[gt_rs > 0] = [255, 0, 0]
        Image.fromarray(overlay_gt).save(OUT / f"{i}_gt_overlay.png")

        if o.get("mask") is not None:
            pred_rs = np.array(Image.fromarray((o["mask"] * 255).astype(np.uint8)).resize(
                (native_u8.shape[1], native_u8.shape[0]), Image.NEAREST))
            overlay_pred = native_u8.copy()
            overlay_pred[pred_rs > 0] = [0, 255, 0]
            Image.fromarray(overlay_pred).save(OUT / f"{i}_pred_overlay.png")
        else:
            print(f"    (未判定为缺陷,没有预测掩膜)", flush=True)

    print(f"图片存到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
