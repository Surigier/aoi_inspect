"""多种子EAD学生集成探针(平常条件唯一未试的准确度手段:降运行方差=抬现场一次fit的下限)。
每类独立训6个EAD(种子0-5):单模型=s0/s1/s2三样本;集成=(0,1)(2,3)(4,5)三对平均分。
量:图级平衡acc 均值±std(阈值fit标定,EAD-only口径,方差收窄是主目标)+ score延时(1 vs 2模型)。
fit不计时→集成训练代价免费;推理2×前向是唯一代价(可优化共享冻结教师,先测上界)。
用法:PYTHONPATH=. python scripts/run_ead_ensemble.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from aoi.efficientad import EfficientADDetector
from aoi.fewshot import FewShotAdapter
from aoi.imageio import load_fast
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
AD2 = Path("data/mvtec_ad_2")
N_MODELS = 6


def prep_battery():
    d = json.load(open(RJ / "phone_battery.json")); R = RI / "phone_battery"
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_d = [_load_img(R / x["image_path"], 640) for x in ng[:30]]
    test_d = [_load_img(R / x["image_path"], 640) for x in ng[30:70]]
    goods = [_load_img(R / x["image_path"], 640) for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit_d, test_d, goods


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_d = [_load_img(R / x["image_path"], 640) for x in ng[:30]]
    test_d = [_load_img(R / x["image_path"], 640) for x in ng[30:70]]
    goods = [_load_img(R / x["image_path"], 640) for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit_d, test_d, goods


def prep_hazelnut():
    root = Path("data/mvtec/hazelnut")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    defs = []
    for fo in ["crack", "cut", "hole"]:
        defs += sorted(glob.glob(str(root / "test" / fo / "*.png")))
    random.Random(0).shuffle(defs)
    fit_d = [_load_img(p, 640) for p in defs[:30]]
    test_d = [_load_img(p, 640) for p in defs[30:70]]
    goods = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    return normals, fit_d, test_d, goods


def prep_sheet():
    root = AD2 / "sheet_metal"
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    fit_d = [load_fast(p) for p in bad[:30]]
    test_d = [load_fast(p) for p in bad[30:70]]
    goods = [load_fast(p) for p in sorted(glob.glob(str(root / "test_public/good/*.png")))[:40]]
    return normals, fit_d, test_d, goods


def bal_acc(fn, fd, td, tg):
    thr = FewShotAdapter._calibrate(list(fn), list(fd))
    rec = float(np.mean([s >= thr for s in td]))
    nacc = float(np.mean([s < thr for s in tg]))
    return (rec + nacc) / 2, rec, nacc


def run(name, prep):
    normals, fit_d, test_d, goods = prep()
    # 6个独立EAD + 各自打分缓存
    SC = []                                                    # 每模型 {fn,fd,td,tg}
    for seed in range(N_MODELS):
        torch.manual_seed(seed)
        det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
        det.fit_fewshot(normals, None)
        sc = {"fn": [det._image_score(x)[0] for x in normals],
              "fd": [det._image_score(x)[0] for x in fit_d],
              "td": [det._image_score(x)[0] for x in test_d],
              "tg": [det._image_score(x)[0] for x in goods]}
        SC.append(sc)
        if seed == 0:                                          # 延时:单模型 vs 双模型顺序前向
            img = test_d[0]
            for _ in range(3):
                det._image_score(img)
            t0 = time.perf_counter()
            for _ in range(10):
                det._image_score(img)
            t1 = (time.perf_counter() - t0) / 10 * 1000
            print(f"{name}: 单模型score={t1:.0f}ms → 双模型集成≈{2*t1:.0f}ms(上界,可共享教师优化)", flush=True)
        print(f"  {name} seed{seed} 训完", flush=True)
    # 单模型三样本(s0/s1/s2)
    singles = [bal_acc(SC[i]["fn"], SC[i]["fd"], SC[i]["td"], SC[i]["tg"]) for i in range(3)]
    # 双模型集成三样本((0,1)(2,3)(4,5),平均原始分再标定)
    pairs = [(0, 1), (2, 3), (4, 5)]
    ens = []
    for a, b in pairs:
        avg = lambda k: [(x + y) / 2 for x, y in zip(SC[a][k], SC[b][k])]
        ens.append(bal_acc(avg("fn"), avg("fd"), avg("td"), avg("tg")))
    s = np.array([x[0] for x in singles]); e = np.array([x[0] for x in ens])
    print(f"{name:12s} 单模型 平衡acc={s.mean():.3f}±{s.std():.3f} (min={s.min():.3f})  |  "
          f"双集成 平衡acc={e.mean():.3f}±{e.std():.3f} (min={e.min():.3f})  "
          f"Δ均值={e.mean()-s.mean():+.3f} Δ下限={e.min()-s.min():+.3f}", flush=True)


def main():
    import sys
    print("=== 多种子EAD集成:单模型 vs 双集成(平衡acc 均值±std/下限,EAD-only口径)===", flush=True)
    jobs = {"battery": prep_battery, "sheet_metal": prep_sheet,
            "pcb": lambda: prep_realiad("pcb"), "hazelnut": prep_hazelnut}
    for name in (sys.argv[1:] or ["battery", "sheet_metal"]):
        run(name, jobs[name])


if __name__ == "__main__":
    main()
