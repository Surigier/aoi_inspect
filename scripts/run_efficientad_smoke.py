"""EfficientAD 地基生死验证:python scripts/run_efficientad_smoke.py [类别...]
1) 真100张正常训练 → AUROC(精度) + 单图延时
2) 8张 vs 100张 → 延时应【恒定】(无记忆库的核心卖点)
3) 不同 train_steps 看少样本训练可行性"""
import sys
import time
import random
import torch
from aoi.efficientad import EfficientADDetector
from aoi.fusion import auroc
from eval.mvtec import load_category

CATS = ["bottle", "screw"]


def run(cat, n_fit, steps, dev):
    data = load_category(f"data/mvtec/{cat}", size=256)
    rng = random.Random(0)
    nm = data["train_normal"][:]; rng.shuffle(nm)
    fn = nm[:n_fit]
    fd = data["test_defect"][:30]
    test = data["test_normal"] + data["test_defect"]
    lab = [0] * len(data["test_normal"]) + [1] * len(data["test_defect"])
    t = time.time()
    det = EfficientADDetector(model_size="small", device=dev, train_steps=steps)
    det.fit_fewshot(fn, fd)
    fit_s = time.time() - t
    scores = [det._image_score(im)[0] for im in test]
    det.predict(test[0])
    lat = sum(det.predict(test[i])["latency_ms"] for i in range(min(5, len(test)))) / min(5, len(test))
    au = auroc(scores, lab)
    print(f"{cat:8s} fit={n_fit:3d}张 steps={steps:5d} AUROC={au:.3f} 延时={lat:.0f}ms fit用时={fit_s:.0f}s", flush=True)
    return au, lat


def main():
    cats = sys.argv[1:] or CATS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备={dev}", flush=True)
    for cat in cats:
        run(cat, 100, 4000, dev)
        run(cat, 8, 4000, dev)        # 对比:延时应与100张几乎相同(无库→恒定)
    print("\n关键看:① AUROC 是否够用 ② 8张vs100张延时是否恒定(证明无库地基)")


if __name__ == "__main__":
    main()
