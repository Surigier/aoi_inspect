"""优先级1验证:OOF阈值+4头bagging+soft-loss(新seg_head.py)vs 双头集成+pooled-F1阈值
(旧seg_head.py,commit ae5fbbb,存档为aoi/_seg_head_old_ae5fbbb.py)真实数据A/B。
只测seg_head本身(WRN特征提取+头训练+阈值标定),不跑EAD/DINO/SAM/fit_fewshot整链——
两版接口都只在extractor=None时才碰det,这里显式传extractor绕开,省掉~15分钟/类的
无关开销(EAD训练+DINO/SAM标定与seg_head精度无关)。
用法:PYTHONPATH=. python scripts/run_seg_head_ab.py
"""
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead as NewHead, _per_image_iou, _mask_to
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as OldHead
from aoi.imageio import load_fast

AD2 = Path("data/mvtec_ad_2")
HW = (256, 256)
SEG_IN = 512


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def prep_ad2(cat, n_norm=100, n_fit=30, n_test=40):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    m = lambda p: _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test_defs = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + n_test]]
    return normals, fit_i, fit_m, test_defs


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(layers=(1, 2), device=dev)

    @torch.no_grad()
    def extractor(img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(dev)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]

    cats = ["sheet_metal", "walnuts", "fruit_jelly"]
    results = {}
    for cat in cats:
        normals, fit_i, fit_m, test_defs = prep_ad2(cat)
        old = OldHead(device=dev, extractor=extractor)
        new = NewHead(device=dev, extractor=extractor)
        ok_old = old.fit(None, fit_i, fit_m, normals[:30])
        ok_new = new.fit(None, fit_i, fit_m, normals[:30])
        print(f"{cat}: old.fit={ok_old} thr={getattr(old,'thr',None)}  "
              f"new.fit={ok_new} head_kind={getattr(new,'head_kind',None)} "
              f"thr_iou={getattr(new,'thr_iou',None)} thr_boxhit={getattr(new,'thr_boxhit',None)}", flush=True)

        def eval_head(head, use_thr):
            ious = []
            for img, gt in test_defs:
                amap = head.map(None, img, HW)
                if amap is None or use_thr is None:
                    ious.append(0.0); continue
                pred = (amap >= use_thr).astype(np.uint8)
                TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(ious))

        old_iou = eval_head(old, getattr(old, "thr", None))
        new_iou_default = eval_head(new, getattr(new, "thr", None))          # 新默认阈值(thr_iou)
        new_iou_boxhit = eval_head(new, getattr(new, "thr_boxhit", None))    # 新框命中阈值(供参考)
        results[cat] = (old_iou, new_iou_default, new_iou_boxhit)
        print(f"  {cat:14s} old(双头+pooled-F1)={old_iou:.3f}  new(4头bagging+OOF-IoU阈值)={new_iou_default:.3f}  "
              f"Δ={new_iou_default-old_iou:+.3f}  new(OOF-boxhit阈值,仅供参考)={new_iou_boxhit:.3f}", flush=True)

    print("\n=== 均值 ===")
    o = np.mean([v[0] for v in results.values()])
    n = np.mean([v[1] for v in results.values()])
    print(f"old={o:.3f}  new={n:.3f}  Δ={n-o:+.3f}", flush=True)


if __name__ == "__main__":
    main()
