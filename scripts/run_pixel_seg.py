"""像素级分割评测(赛题按定位评准确率):EAD 异常图 vs GT 掩膜 → pixel-AUROC。
迁移=100 正常 fit(EAD 无监督);测试缺陷+正常图逐像素打分,对 GT 掩膜算 pixel-AUROC。
评测在 EVAL_HW 分辨率(降采样统一,标准做法,控算量)。
用法:python scripts/run_pixel_seg.py [mvtec|realiad|both]
"""
import sys
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from eval.protocol import image_auroc
from eval.mvtec import _load_img

EVAL_HW = (256, 256)
N_FIT = 100


def _mask(path, hw):
    if path is None or not Path(path).exists():
        return np.zeros(hw, dtype=np.uint8)
    m = Image.open(path).convert("L").resize((hw[1], hw[0]))
    return (np.array(m) > 0).astype(np.uint8)


def eval_category(name, fit_normals, test_items):
    """test_items: [(img_tensor_CHW, mask_path_or_None), ...];正常图 mask=None。"""
    det = EfficientADDetector(model_size="small", device="cuda" if torch.cuda.is_available() else "cpu",
                              train_steps=8000)
    det.fit_fewshot(fit_normals, None)
    ps, ls = [], []
    for img, mpath in test_items:
        amap = det.anomaly_map_large(img, out_hw=EVAL_HW)
        gt = _mask(mpath, EVAL_HW)
        ps.append(amap.ravel()); ls.append(gt.ravel())
    scores = np.concatenate(ps); labels = np.concatenate(ls)
    # 像素级 AUROC(子采样控算量,正负各取上限)
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    rng = np.random.RandomState(0)
    if len(pos) > 200000: pos = rng.choice(pos, 200000, replace=False)
    if len(neg) > 200000: neg = rng.choice(neg, 200000, replace=False)
    idx = np.concatenate([pos, neg])
    au = image_auroc(scores[idx], labels[idx])
    print(f"{name:18s} pixel-AUROC={au:.3f}  (缺陷像素占比={labels.mean()*100:.2f}%, n图={len(test_items)})", flush=True)
    return au


def run_mvtec(cats=("bottle", "metal_nut", "cable", "transistor", "screw")):
    """官方数据集 MVTec AD。掩膜在 _gt_stage(因 data/mvtec 不可写未就地移动)。"""
    GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
    aus = []
    for c in cats:
        root = Path(f"data/mvtec/{c}")
        gn = sorted(glob.glob(str(root / "train/good/*.png")))[:N_FIT]
        if not gn:
            print(f"mvtec/{c}: 无数据"); continue
        fit = [_load_img(p, 320) for p in gn]
        items = []
        for dt in sorted(glob.glob(str(root / "test/*"))):
            dtn = Path(dt).name
            for p in sorted(glob.glob(f"{dt}/*.png")):
                if dtn == "good":
                    items.append((_load_img(p, 320), None))
                else:
                    mp = str(GT / c / "ground_truth" / dtn / (Path(p).stem + "_mask.png"))
                    items.append((_load_img(p, 320), mp))
        aus.append(eval_category(f"mvtec/{c}", fit, items))
    return aus


def run_visa(cats=("pcb1", "pcb2")):
    aus = []
    for c in cats:
        base = Path(f"data/visa/{c}/Data")
        gn = sorted(glob.glob(str(base / "Images/Normal/*.JPG")))
        if not gn:
            print(f"visa/{c}: 无数据"); continue
        random.Random(0).shuffle(gn)
        fit = [_load_img(p, 320) for p in gn[:N_FIT]]
        items = [(_load_img(p, 320), None) for p in gn[N_FIT:N_FIT + 200]]
        for p in sorted(glob.glob(str(base / "Images/Anomaly/*.JPG"))):
            mp = str(base / "Masks/Anomaly" / (Path(p).stem + ".png"))
            items.append((_load_img(p, 320), mp))
        aus.append(eval_category(f"visa/{c}", fit, items))
    return aus


def run_ad2(cats=("sheet_metal", "can")):
    aus = []
    for c in cats:
        root = Path(f"data/mvtec_ad_2/{c}")
        gn = sorted(glob.glob(str(root / "train/good/*.png")))[:N_FIT]
        if not gn:
            print(f"ad2/{c}: 无数据"); continue
        fit = [_load_img(p, 512) for p in gn]                  # AD2 大图,稍高分辨率
        items = [(_load_img(p, 512), None)
                 for p in sorted(glob.glob(str(root / "test_public/good/*.png")))[:100]]
        for p in sorted(glob.glob(str(root / "test_public/bad/*.png"))):
            mp = str(root / "test_public/ground_truth/bad" / (Path(p).stem + "_mask.png"))
            items.append((_load_img(p, 512), mp))
        aus.append(eval_category(f"ad2/{c}", fit, items))
    return aus


def run_realiad(cats=("pcb", "phone_battery", "sim_card_set")):
    ROOT = Path("data/_dl/Real-IAD")
    JD = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
    aus = []
    for c in cats:
        d = json.load(open(JD / f"{c}.json"))
        train_ok = [it for it in d["train"] if it["anomaly_class"] == "OK"]
        rng = random.Random(0); rng.shuffle(train_ok)
        fit = [_load_img(ROOT / c / it["image_path"], 320) for it in train_ok[:N_FIT]]
        items = []
        for it in d["test"]:
            img = _load_img(ROOT / c / it["image_path"], 320)
            mp = str(ROOT / c / it["mask_path"]) if it.get("mask_path") else None
            items.append((img, mp))
        aus.append(eval_category(f"realiad/{c}", fit, items))
    return aus


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    torch.manual_seed(0)
    aus = []
    if mode in ("mvtec", "all"): aus += run_mvtec()
    if mode in ("visa", "all"): aus += run_visa()
    if mode in ("ad2", "all"): aus += run_ad2()
    if mode in ("realiad", "all"): aus += run_realiad()
    if aus:
        print(f"\n像素级 pixel-AUROC 均值({len(aus)}类): {sum(aus)/len(aus):.3f}")


if __name__ == "__main__":
    main()
