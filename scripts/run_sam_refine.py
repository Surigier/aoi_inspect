"""SAM边界精化实验(解决定位IoU:粗定位→SAM出锐利边界)。
思路=Segment Any Anomaly(SAA+)同族:异常图连通域→框提示→MobileSAM精化掩膜。
防爆保护:SAM掩膜面积>提示框4倍(分割了整个物体而非缺陷)→回退原掩膜。
量:逐图IoU 原始vs+SAM,以及SAM延时开销。用法:python scripts/run_sam_refine.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)
SAM_SZ = 512


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou(pred, gt):
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int(((~pred.astype(bool)) & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


class SamRefiner:
    def __init__(self):
        from ultralytics import SAM
        self.m = SAM("mobile_sam.pt")

    def refine(self, img_chw, raw_mask):
        """img(3,H,W)[0,1] tensor + 原始二值掩膜(256²) → SAM精化掩膜(256²)。
        每个连通域→框提示(pad 15%);掩膜面积>框4倍→回退该区域原掩膜。"""
        H, W = raw_mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return raw_mask
        arr = (img_chw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ih, iw = arr.shape[:2]
        sx, sy = iw / W, ih / H
        boxes = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 3:
                continue
            px, py = max(2, int(w * 0.15)), max(2, int(h * 0.15))
            boxes.append([max(0, (x - px) * sx), max(0, (y - py) * sy),
                          min(iw, (x + w + px) * sx), min(ih, (y + h + py) * sy)])
        if not boxes:
            return raw_mask
        r = self.m.predict(arr, bboxes=boxes, imgsz=SAM_SZ, verbose=False)[0]
        if r.masks is None:
            return raw_mask
        out = np.zeros((H, W), np.uint8)
        for k, b in enumerate(boxes):
            m = r.masks.data[k].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            bx = [int(b[0] / sx), int(b[1] / sy), int(b[2] / sx), int(b[3] / sy)]
            box_area = max(1, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            if m.sum() > 4 * box_area or m.sum() == 0:          # SAM爆了/空→回退原区域
                out[bx[1]:bx[3], bx[0]:bx[2]] |= raw_mask[bx[1]:bx[3], bx[0]:bx[2]]
            else:
                out |= m
        return out


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [_load_img(p, 320) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW) for p, fo in df[:k]]
    tests = [(_load_img(p, 320), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return normals, fit_i, fit_m, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_i = [_load_img(R / x["image_path"], 320) for x in ng[:30]]
    fit_m = [_read(R / x["mask_path"], HW) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    return normals, fit_i, fit_m, tests


def main():
    torch.manual_seed(0)
    sam = SamRefiner()
    print(f"=== SAM边界精化 × 逐图IoU(imgsz={SAM_SZ})===")
    jobs = [
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
    ]
    R0, R1 = [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, tests = prep()
        det = CompetitionLargeDetector(train_steps=200)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        i_raw, i_sam, ts = [], [], []
        for img, gt in tests:
            amap = det.segment(img)
            raw = (amap >= det.pix_thr).astype(np.uint8)
            i_raw.append(iou(raw.astype(bool), gt))
            t0 = time.perf_counter()
            ref = sam.refine(img, raw)
            ts.append((time.perf_counter() - t0) * 1000)
            i_sam.append(iou(ref.astype(bool), gt))
        R0.append(np.mean(i_raw)); R1.append(np.mean(i_sam))
        print(f"{name:18s} 原始IoU={np.mean(i_raw):.3f}  +SAM={np.mean(i_sam):.3f}  Δ={np.mean(i_sam)-np.mean(i_raw):+.3f}  SAM开销={np.mean(ts):.0f}ms", flush=True)
    print(f"\n均值: 原始={np.mean(R0):.3f}  +SAM={np.mean(R1):.3f}  Δ={np.mean(R1)-np.mean(R0):+.3f}")


if __name__ == "__main__":
    main()
