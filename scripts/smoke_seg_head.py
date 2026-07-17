"""极小冒烟测试:验证新seg_head.py(soft target+bagging+OOF阈值)跑通不崩,不追求精度。
用法:PYTHONPATH=. python scripts/smoke_seg_head.py
"""
import glob
import random
import numpy as np
import torch
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead
from eval.mvtec import _load_img
from pathlib import Path
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    root = Path("data/mvtec/hazelnut")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:10]]
    gt_root = Path("data/_dl/_gt_stage/mvtech_anomaly_detection/hazelnut/ground_truth/crack")
    defs = sorted(glob.glob(str(root / "test/crack/*.png")))[:8]
    def_imgs = [_load_img(p, 320) for p in defs]
    def_masks = []
    for p in defs:
        mp = gt_root / (Path(p).stem + "_mask.png")
        if mp.exists():
            m = (np.array(Image.open(mp).convert("L").resize((320, 320))) > 0).astype(np.uint8)
        else:
            m = np.zeros((320, 320), np.uint8)
        def_masks.append(m)

    bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)

    def extractor(img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        import torch.nn.functional as F
        x = F.interpolate(x, size=(320, 320), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]

    head = SupervisedSegHead(device=DEV, steps=30, extractor=extractor)   # steps小,只测跑通
    ok = head.fit(None, def_imgs, def_masks, normals)
    print(f"fit返回: {ok}")
    print(f"head_kind: {getattr(head, 'head_kind', None)}")
    print(f"thr={head.thr} thr_iou={head.thr_iou} thr_boxhit={head.thr_boxhit} thr_f1={head.thr_f1}")
    amap = head.map(None, def_imgs[0], (256, 256))
    print(f"map输出shape={amap.shape if amap is not None else None} "
          f"范围=[{amap.min():.3f},{amap.max():.3f}]" if amap is not None else "map返回None")
    print("=== seg_head冒烟测试通过 ===")

    # SAM受控精化冒烟(小规模)
    from aoi.sam_refine import SamRefiner
    sam = SamRefiner()

    def seg_map_fn(img):
        return head.map(None, img, (256, 256))

    sam.calibrate(type("D", (), {"seg_head": head})(), def_imgs, def_masks, seg_map_fn, k=2)
    print(f"SAM校准: padding={sam.padding} gate={sam.gate} gain={getattr(sam,'calib_gain',None)}")
    raw_amap = seg_map_fn(def_imgs[0])
    raw_mask = (raw_amap >= head.thr).astype(np.uint8)
    refined = sam.refine(def_imgs[0], raw_mask, amap=raw_amap)
    print(f"refine输出shape={refined.shape} 和 raw差异像素数={int((refined!=raw_mask).sum())}")
    print("=== SAM冒烟测试通过,无崩溃 ===")


if __name__ == "__main__":
    main()
