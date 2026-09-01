"""挑我们已验证过表现最好的几个类目+leon的phone_best混合,带掩膜走监督定位路径。
明确说明:这是选数据,不是算法创新——hazelnut(单独0.950)/pill(历史1.000)/
sim_card_set(Real-IAD 0.925)/phone_battery(Real-IAD 0.912~0.925)都是本项目
已验证的强类目,phone_best是leon要求必须包含的手机屏数据。per_mode_gate维持
默认关(已两次判负,不再重开)。

用法:PYTHONPATH=. python scripts/run_best_mix.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
import make_exam_data as m
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import box_hit, gt_boxes

RNG = random.Random(0)
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")


def mvtec_pool(cat):
    root = Path(f"data/mvtec/{cat}")
    normals = sorted(root.glob("train/good/*.png"))
    ok_test = sorted(root.glob("test/good/*.png"))
    defs = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            for f in sorted(sub.glob("*.png")):
                mk = GT / cat / "ground_truth" / sub.name / (f.stem + "_mask.png")
                if mk.exists():
                    defs.append((f, mk))
    for lst in (normals, defs, ok_test):
        RNG.shuffle(lst)
    return normals, defs, ok_test


def realiad_pool(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ok_train = [x for x in d["train"] if x["anomaly_class"] == "OK"]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
    ok_test = [x for x in d["test"] if x["anomaly_class"] == "OK"]
    for lst in (ok_train, ng, ok_test):
        RNG.shuffle(lst)
    normals = [R / x["image_path"] for x in ok_train]
    defs = [(R / x["image_path"], R / x["mask_path"]) for x in ng]
    goods = [R / x["image_path"] for x in ok_test]
    return normals, defs, goods


def phone_pool():
    msd = sorted(m.MSD_GOOD.glob("*.png")); RNG.shuffle(msd)
    pb = m.phone_best_defects(6 + 500)
    normals = msd            # 全部20张,官方硬约束,不复制凑数
    defs = [(f, mk) for f, mk, _ in pb]   # mk已经是(h,w) ndarray,不是路径
    return normals, defs, []


def load_mask(mk, native_hw=None):
    if isinstance(mk, np.ndarray):
        return (mk > 0).astype(np.uint8)
    return (np.array(Image.open(mk).convert("L")) > 0).astype(np.uint8)


def main(seg_gate=False):
    pools = {
        "hazelnut": mvtec_pool("hazelnut"),
        "pill": mvtec_pool("pill"),
        "sim_card_set": realiad_pool("sim_card_set"),
        "phone_battery": realiad_pool("phone_battery"),
        "phone_best": phone_pool(),
    }
    CATS = list(pools)

    fit_normals, fit_defects, fit_masks = [], [], []
    for c in CATS:
        norm, defs, _ = pools[c]
        n_norm = 20 if c != "phone_best" else len(norm)   # phone_best全用,其余20
        for p in norm[:n_norm]:
            fit_normals.append(load_fast(p))
        for p, mk in defs[:6]:
            fit_defects.append(load_fast(p))
            fit_masks.append(load_mask(mk))
    print(f"fit: 正常{len(fit_normals)} 缺陷{len(fit_defects)}(5类,**带掩膜**,per_mode_gate=False,seg_gate={seg_gate})", flush=True)

    det = CompetitionLargeDetector(compile_infer=True, seg_gate=seg_gate)
    det.fit_fewshot(fit_normals, fit_defects, defect_masks=fit_masks)
    print(f"fit完成,阈值={det.decision_threshold():.4f}", flush=True)
    del fit_normals, fit_defects, fit_masks

    pool = []
    for c in CATS:
        norm, defs, goods = pools[c]
        used_norm = 20 if c != "phone_best" else len(norm)
        for p, mk in defs[6:]:
            pool.append((p, mk, "缺陷", c))
        for p in (goods + norm[used_norm:])[:150]:
            pool.append((p, None, "正常", c))
    RNG.shuffle(pool)

    tp = fp = fn = tn = 0; hits = []
    by_cat = {}
    for idx, (p, mk, truth, cat) in enumerate(pool):
        img = load_fast(p)
        o = det.locate(img)
        pred = bool(o["is_defect"])
        is_def = truth == "缺陷"
        ok = (is_def == pred)
        if is_def and pred: tp += 1
        elif is_def: fn += 1
        elif pred: fp += 1
        else: tn += 1
        by_cat.setdefault(cat, [0, 0]); by_cat[cat][0] += ok; by_cat[cat][1] += 1
        if is_def:
            gm = load_mask(mk)
            gbs = gt_boxes(gm)
            if pred and o.get("boxes") and gbs:
                h = box_hit(o["boxes"], gbs)
                hits.append(h if h is not None else 0.0)
            else:
                hits.append(0.0)
        if (idx + 1) % 200 == 0:
            n = idx + 1
            print(f"  已测 {n}/{len(pool)} acc={(tp+tn)/n:.3f}", flush=True)

    n = tp + fp + fn + tn
    print(f"\n=== 最好类目混合(带掩膜,per_mode_gate=False,seg_gate={seg_gate}) ===", flush=True)
    print(f"n={n} acc={(tp+tn)/n:.3f} 召回={tp/max(tp+fn,1):.3f} 误报率={fp/max(fp+tn,1):.3f} "
          f"框命中@0.5={np.mean(hits):.3f}", flush=True)
    print(f"RESULT seg_gate={seg_gate} acc={(tp+tn)/n:.4f} recall={tp/max(tp+fn,1):.4f} "
          f"fpr={fp/max(fp+tn,1):.4f} hit={np.mean(hits):.4f}", flush=True)
    for c, (ok, tot) in sorted(by_cat.items()):
        print(f"  {c:16s} n={tot:4d} acc={ok/tot:.3f}", flush=True)


if __name__ == "__main__":
    main(seg_gate="--seg-gate" in sys.argv)
