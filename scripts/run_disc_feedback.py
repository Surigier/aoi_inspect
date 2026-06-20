"""判别头·漏检反馈模型级验证:python scripts/run_disc_feedback.py [pcb3]
弱起步(100正常+5缺陷)训判别头 → 每轮反馈漏检缺陷 → 重训(改模型参数)
→ 测"固定误报率下的召回(recall@FPR=10%,阈值无关)"。若上升,说明模型本身变好(非只动阈值)。"""
import sys
import random
from pathlib import Path
import numpy as np
import torch
from aoi.backbone import Backbone
from aoi.branches.disc_ad import DiscriminativeBranch
from eval.mvtec import _load_img


def load_visa(cat, size=320):
    base = Path("data/visa") / cat / "Data" / "Images"
    n = [_load_img(p, size) for p in sorted((base / "Normal").glob("*.JPG"))]
    a = [_load_img(p, size) for p in sorted((base / "Anomaly").glob("*.JPG"))]
    return n, a


def recall_at_fpr(br, test_n, test_d, fpr=0.10):
    sn = np.array([br.infer(x.unsqueeze(0)).score for x in test_n])
    sd = np.array([br.infer(x.unsqueeze(0)).score for x in test_d])
    thr = np.quantile(sn, 1 - fpr)           # 固定 10% 误报对应的阈值
    return float((sd >= thr).mean())


def main(cat="pcb3"):
    torch.manual_seed(0)
    normal, anom = load_visa(cat)
    rng = random.Random(0)
    rng.shuffle(normal)
    rng.shuffle(anom)
    fit_n, test_n = normal[:100], normal[100:300]
    defects, fb_d, test_d = list(anom[:5]), anom[5:60], anom[60:]
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    br = DiscriminativeBranch(backbone=bb, epochs=300, lr=0.05)
    br.fit_supervised(fit_n, defects)
    print(f"visa/{cat}  起始(100正常+5缺陷)  recall@10%FP={recall_at_fpr(br, test_n, test_d):.3f}  缺陷={len(defects)}")
    for rnd in range(1, 5):
        thr_now = np.quantile([br.infer(x.unsqueeze(0)).score for x in test_n], 0.9)
        missed = [x for x in fb_d if br.infer(x.unsqueeze(0)).score < thr_now][:15]
        if not missed:
            break
        ids = {id(x) for x in missed}
        defects += missed
        fb_d = [x for x in fb_d if id(x) not in ids]
        br.fit_supervised(fit_n, defects)        # 重训 = 动态调整模型参数
        print(f"  反馈{len(missed)}张漏检后  recall@10%FP={recall_at_fpr(br, test_n, test_d):.3f}  缺陷={len(defects)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pcb3")
