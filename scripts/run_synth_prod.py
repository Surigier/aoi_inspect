"""验证生产 SupervisedSegHead 的合成增广(n_synth):同域 pixel-AUROC,0 vs 60 对比。
确认集成正确 + 同域不掉(跨域增益已在 run_synth_aug.py 验证 +0.065)。
用法:python scripts/run_synth_prod.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from aoi.seg_head import SupervisedSegHead
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256)


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def items_mvtec(cat):
    root = Path(f"data/mvtec/{cat}")
    norm = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    defs, tests = [], []
    for dt in sorted(glob.glob(str(root / "test/*"))):
        dtn = Path(dt).name
        for p in sorted(glob.glob(f"{dt}/*.png")):
            if dtn == "good":
                tests.append((_load_img(p, 320), None))
            else:
                defs.append((_load_img(p, 320), str(GT / cat / "ground_truth" / dtn / (Path(p).stem + "_mask.png"))))
    random.Random(0).shuffle(defs)
    return norm, defs[:30], tests + defs[30:]


def items_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    norm = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    defs = [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), None) for x in d["test"] if x["anomaly_class"] == "OK"][:150]
    tests += [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[30:70]]
    return norm, defs, tests


def evalh(det, head, tests):
    S, L = [], []
    for img, mp in tests:
        amap = head.map(det, img, EVAL_HW)
        S.append(amap.ravel()); L.append(_mask(mp, EVAL_HW).ravel())
    return image_auroc(np.concatenate(S), np.concatenate(L))


def main():
    torch.manual_seed(0)
    jobs = [("mvtec/transistor", items_mvtec, "transistor"),
            ("realiad/pcb", items_realiad, "pcb"),
            ("realiad/phone_battery", items_realiad, "phone_battery")]
    for name, fn, cat in jobs:
        norm, defs, tests = fn(cat)
        det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
        det.fit_fewshot(norm, None)
        d_imgs = [d[0] for d in defs]
        d_masks = [_mask(d[1], EVAL_HW) for d in defs]
        row = {}
        for ns in [0, 60]:
            h = SupervisedSegHead(device=DEV, n_synth=ns)
            h.fit(det, d_imgs, d_masks, norm[:30])
            row[ns] = evalh(det, h, tests)
        print(f"{name:22s} 无合成={row[0]:.3f}  +合成60={row[60]:.3f}  Δ={row[60]-row[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
