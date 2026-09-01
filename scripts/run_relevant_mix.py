"""重新混合,贴近赛题隐藏域(手机部件):全部换成Real-IAD真实电子/手机部件类目
(phone_battery/sim_card_set/pcb/usb_adaptor/audiojack),**不再用MVTec的坚果/
线缆/药片**这类和"手机部件"毫无关系的日用品凑数——这5个类目本身就是Real-IAD
12类目成绩单里验证过的真实数据,难度也有高有低(phone_battery/sim_card_set好、
pcb中等、usb_adaptor/audiojack弱),比强行拉MVTec进来更贴近赛题描述。

按leon要求:①一次混合fit(100正常+30缺陷,5类均摊)②**不传掩膜**(测试无监督定位
路径,不训监督分割头)③per_mode_gate开关做A/B(混域阈值失配的候选修复)。

用法:PYTHONPATH=. python scripts/run_relevant_mix.py [--per-mode]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import box_hit, gt_boxes

RNG = random.Random(0)
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")


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


def main(per_mode=False):
    pools = {
        "phone_battery": realiad_pool("phone_battery"),
        "sim_card_set": realiad_pool("sim_card_set"),
        "pcb": realiad_pool("pcb"),
        "usb_adaptor": realiad_pool("usb_adaptor"),
        "audiojack": realiad_pool("audiojack"),
    }
    CATS = list(pools)

    fit_normals, fit_defects = [], []
    for c in CATS:
        norm, defs, _ = pools[c]
        fit_normals += [load_fast(p) for p in norm[:20]]
        fit_defects += [load_fast(p) for p, _ in defs[:6]]
    print(f"fit: 正常{len(fit_normals)} 缺陷{len(fit_defects)}(5类均摊,**不传掩膜**)", flush=True)

    det = CompetitionLargeDetector(per_mode_gate=per_mode, compile_infer=True)
    det.fit_fewshot(fit_normals, fit_defects, defect_masks=None)
    print(f"fit完成,per_mode_gate={per_mode},阈值={det.decision_threshold():.4f}", flush=True)
    del fit_normals, fit_defects

    pool = []
    for c in CATS:
        norm, defs, goods = pools[c]
        for p, mk in defs[6:]:
            pool.append((p, mk, "缺陷", c))
        # Real-IAD每类目正常图上千张,全用测试时间爆炸——每类目正常测试图封顶150张,
        # 和赛题"1000+张混合流"同一量级,不是要把每类目全部测完
        for p in (goods + norm[20:])[:150]:
            pool.append((p, None, "正常", c))
    RNG.shuffle(pool)

    tp = fp = fn = tn = 0; hits = []
    for idx, (p, mk, truth, cat) in enumerate(pool):
        img = load_fast(p)
        o = det.locate(img)
        pred = bool(o["is_defect"])
        is_def = truth == "缺陷"
        if is_def and pred: tp += 1
        elif is_def: fn += 1
        elif pred: fp += 1
        else: tn += 1
        if is_def:
            from PIL import Image
            gm = (np.array(Image.open(mk).convert("L")) > 0).astype("uint8")
            gbs = gt_boxes(gm)
            if pred and o.get("boxes"):
                h = box_hit(o["boxes"], gbs)
                hits.append(h if h is not None else 0.0)
            else:
                hits.append(0.0)
        if (idx + 1) % 200 == 0:
            n = idx + 1
            print(f"  已测 {n}/{len(pool)} acc={(tp+tn)/n:.3f}", flush=True)

    n = tp + fp + fn + tn
    print(f"\n=== per_mode_gate={per_mode} ===", flush=True)
    print(f"n={n} acc={(tp+tn)/n:.3f} 召回={tp/max(tp+fn,1):.3f} 误报率={fp/max(fp+tn,1):.3f} "
          f"框命中@0.5={np.mean(hits):.3f}", flush=True)
    print(f"RESULT per_mode={per_mode} acc={(tp+tn)/n:.4f} recall={tp/max(tp+fn,1):.4f} "
          f"fpr={fp/max(fp+tn,1):.4f} hit={np.mean(hits):.4f}", flush=True)


if __name__ == "__main__":
    main(per_mode="--per-mode" in sys.argv)
