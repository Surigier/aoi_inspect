"""Real-IAD 手机件 × 比赛实际大图检测器(CompetitionLargeDetector=EAD)。
回答关键问题:比赛在2500²手机图上实际跑的是EAD大图路径(非default_adapter)——
EAD在手机件域上行不行?把'域'与'分辨率'两轴拆开(AD2已验分辨率,此处验域)。
用法:python scripts/run_realiad_ead.py [upscale]   (upscale=放大到~2000²模拟大图)
"""
import sys
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from aoi.competition import CompetitionLargeDetector
from aoi.fusion import auroc
from eval.mvtec import _load_img

JSON_DIR = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
IMG_ROOT = Path("data/_dl/Real-IAD")
CATS = ["phone_battery", "pcb", "sim_card_set", "usb", "button_battery"]
UPSCALE = len(sys.argv) > 1 and sys.argv[1] == "upscale"
SIZE = 2000 if UPSCALE else 256


def load_img(cat, rel):
    p = IMG_ROOT / cat / rel
    img = _load_img(p, SIZE)            # (3,SIZE,SIZE) in [0,1]
    return img


def main():
    print(f"=== EAD大图检测器 × Real-IAD手机件 (input={SIZE}{'(放大模拟2500²)' if UPSCALE else '原生'}) ===")
    aus, accs, lats = [], [], []
    for cat in CATS:
        d = json.load(open(JSON_DIR / f"{cat}.json"))
        train_ok = [it for it in d["train"] if it["anomaly_class"] == "OK"]
        test_ok = [it for it in d["test"] if it["anomaly_class"] == "OK"]
        test_ng = [it for it in d["test"] if it["anomaly_class"] != "OK"]
        rng = random.Random(0); rng.shuffle(train_ok); rng.shuffle(test_ng)
        fn = [load_img(cat, it["image_path"]) for it in train_ok[:100]]
        fd = [load_img(cat, it["image_path"]) for it in test_ng[:30]]
        ev = test_ok + test_ng[30:]                       # 评测:全部正常 + 未用于fit的缺陷
        ev_imgs = [load_img(cat, it["image_path"]) for it in ev]
        ev_lab = [0] * len(test_ok) + [1] * len(test_ng[30:])

        det = CompetitionLargeDetector(train_steps=10000)
        det.fit_fewshot(fn, fd)
        import time
        scores = []
        t0 = time.perf_counter()
        for im in ev_imgs:
            scores.append(det.predict(im)["score"])
        lat = (time.perf_counter() - t0) * 1000.0 / len(ev_imgs)
        au = auroc(scores, ev_lab)
        thr = det.threshold
        acc = sum((s >= thr) == bool(l) for s, l in zip(scores, ev_lab)) / len(ev_lab)
        aus.append(au); accs.append(acc); lats.append(lat)
        print(f"{cat:16s} EAD-AUROC={au:.3f} acc={acc:.3f} lat={lat:.0f}ms (n={len(ev_lab)})", flush=True)
    n = len(aus)
    print(f"\n均值({n}类): EAD-AUROC={sum(aus)/n:.3f} acc={sum(accs)/n:.3f} lat={sum(lats)/n:.0f}ms")


if __name__ == "__main__":
    main()
