"""YOLO-World 零样本部件计数 → 逻辑异常(缺件/数量错)检测 POC。
在 MVTec LOCO breakfast_box / juice_bottle 上跑 good vs logical_anomalies 的 AUROC。
方法:train/good 建每类参考计数分布(中位+MAD)→ test 图算 z-偏离和 → 分数越高越异常。
用法:python scripts/run_yoloworld_loco.py
"""
import glob
import collections
import numpy as np
from ultralytics import YOLOWorld
from aoi.fusion import auroc

# 每个类别的零样本词表(开放词汇,描述场景里应有的部件)
CONFIGS = {
    "pushpins": {
        "vocab": ["yellow push pin"],
        "conf": 0.10,
    },
    "screw_bag": {
        "vocab": ["metal screw bolt", "hex nut", "washer ring"],
        "conf": 0.10,
    },
}
ROOT = "data/_dl/mvtec_loco"
N_REF = 80


def count_vec(model, path, vocab, conf):
    try:
        r = model.predict(path, conf=conf, verbose=False, imgsz=960)[0]
    except Exception:
        return None  # 跳过偶发解码失败
    names = r.names
    c = collections.Counter()
    for b in r.boxes:
        c[names[int(b.cls)]] += 1
    return np.array([c[v] for v in vocab], dtype=float)


def main():
    for cat, cfg in CONFIGS.items():
        vocab, conf = cfg["vocab"], cfg["conf"]
        model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(vocab)

        ref_paths = sorted(glob.glob(f"{ROOT}/{cat}/train/good/*.png"))[:N_REF]
        ref = np.stack([v for p in ref_paths if (v := count_vec(model, p, vocab, conf)) is not None])
        med = np.median(ref, axis=0)
        mad = np.maximum(np.median(np.abs(ref - med), axis=0), 0.5)  # 鲁棒尺度,下限0.5使差1个有意义

        def score(path):
            v = count_vec(model, path, vocab, conf)
            if v is None:
                return 0.0  # 解码失败 → 当正常,不冤枉
            return float(np.sum(np.abs(v - med) / mad))

        good = sorted(glob.glob(f"{ROOT}/{cat}/test/good/*.png"))
        logical = sorted(glob.glob(f"{ROOT}/{cat}/test/logical_anomalies/*.png"))
        struct = sorted(glob.glob(f"{ROOT}/{cat}/test/structural_anomalies/*.png"))

        gs = [score(p) for p in good]
        ls = [score(p) for p in logical]
        ss = [score(p) for p in struct]

        au_log = auroc(gs + ls, [0] * len(gs) + [1] * len(ls))
        au_str = auroc(gs + ss, [0] * len(gs) + [1] * len(ss))
        au_all = auroc(gs + ls + ss, [0] * len(gs) + [1] * (len(ls) + len(ss)))
        print(f"\n=== {cat} | vocab={vocab} conf={conf} ===")
        print(f"  参考计数中位 med={med.tolist()} mad={mad.round(2).tolist()}")
        print(f"  good均分={np.mean(gs):.2f} logical均分={np.mean(ls):.2f} struct均分={np.mean(ss):.2f}")
        print(f"  AUROC  逻辑={au_log:.3f}  结构={au_str:.3f}  全部={au_all:.3f}  (n good={len(gs)} log={len(ls)} str={len(ss)})")


if __name__ == "__main__":
    main()
