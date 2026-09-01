"""诊断:per_mode_gate两次判负(3类MVTec/5类Real-IAD)都指向"DINO CLS聚类分组不准"
这一个根因(过切模态/K定不准)。这里用**已知的真实类别标签**(不是kmeans聚出来的)
直接分组标定,隔离出"分组标定"这个思路本身有没有上限价值——如果连完美分组都救不了
混域,就不用再往"换个更准的聚类算法"这个方向投入;如果完美分组明显更好,才值得继续
想生产环境怎么逼近它(模板匹配等,评委不会给产品身份标签)。

monkey-patch competition._fit_modes/_assign_mode,其余完全走生产 det.locate() 链路
(同 run_best_mix.py 口径),数据也复用同一份5类混合(hazelnut/pill/sim_card_set/
phone_battery/phone_best)。

用法:PYTHONPATH=. python scripts/run_oracle_mode.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import aoi.competition as comp
from run_best_mix import mvtec_pool, realiad_pool, phone_pool, load_mask, RNG
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import box_hit, gt_boxes

from collections import deque

_TRAIN_LABELS = []       # 与fit_fewshot(normals=...)严格同顺序的真实类别
_TRAIN_DEFECT_LABELS = []  # 与fit_fewshot(defects=...)严格同顺序的真实类别
_FIT_QUEUE = deque()     # fit期:_calibrate_dino_gate内部按 k_n(百张normal)→k_d(30张defect)
                          # 顺序连续调用_assign_mode,预先把两段标签接好塞进队列,逐个消费
_TEST_LABEL = [None]     # 测试期(队列已耗尽):当前locate()对应的真实类别
_CATS_ORDER = []


def oracle_fit_modes(C, en, dn, kmax=4, seed=0):
    global _CATS_ORDER, _FIT_QUEUE
    labels = _TRAIN_LABELS
    assert len(labels) == len(en), f"oracle标签数{len(labels)}≠正常图数{len(en)}"
    _CATS_ORDER = sorted(set(labels))
    _FIT_QUEUE = deque(_TRAIN_LABELS + _TRAIN_DEFECT_LABELS)   # 见上,对应k_n+k_d的调用顺序
    X = torch.nn.functional.normalize(C.float(), dim=1)
    modes = []
    for cat in _CATS_ORDER:
        idx = [i for i, l in enumerate(labels) if l == cat]
        c = X[idx].mean(0)
        modes.append(dict(c=c, emu=float(en[idx].mean()), esd=float(en[idx].std() + 1e-9),
                          dmu=float(dn[idx].mean()), dsd=float(dn[idx].std() + 1e-9), n=len(idx)))
    print(f"!! oracle分组(真实标签,非聚类): {[(c, m['n']) for c, m in zip(_CATS_ORDER, modes)]}", flush=True)
    return modes


def oracle_assign_mode(cls_vec, modes):
    """_calibrate_dino_gate内部对k_n/k_d的批量调用,严格按_FIT_QUEUE顺序消费
    (fit期);队列耗尽后(测试期,locate()每次单次调用)改读_TEST_LABEL[0]。
    **不能用cls_vec最近邻猜标签**——那样等于又绕回"聚类式"分组,这次要测的是
    "标签100%已知"这个理想上限,必须是真标签直查。"""
    cat = _FIT_QUEUE.popleft() if _FIT_QUEUE else _TEST_LABEL[0]
    if cat not in _CATS_ORDER:
        return 0
    return _CATS_ORDER.index(cat)


comp._fit_modes = oracle_fit_modes
comp._assign_mode = oracle_assign_mode


def main():
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
        n_norm = 20 if c != "phone_best" else len(norm)
        for p in norm[:n_norm]:
            fit_normals.append(load_fast(p))
            _TRAIN_LABELS.append(c)
        for p, mk in defs[:6]:
            fit_defects.append(load_fast(p))
            fit_masks.append(load_mask(mk))
            _TRAIN_DEFECT_LABELS.append(c)
    print(f"fit: 正常{len(fit_normals)} 缺陷{len(fit_defects)}(5类,oracle真实标签分组)", flush=True)

    det = CompetitionLargeDetector(compile_infer=True, per_mode_gate=True)
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
        _TEST_LABEL[0] = cat                      # 测试图的真实类别,推理时oracle_assign_mode读取
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
    print(f"\n=== oracle真实标签分组(5类混合,per_mode_gate,非聚类) ===", flush=True)
    print(f"n={n} acc={(tp+tn)/n:.3f} 召回={tp/max(tp+fn,1):.3f} 误报率={fp/max(fp+tn,1):.3f} "
          f"框命中@0.5={np.mean(hits):.3f}", flush=True)
    print(f"RESULT oracle_mode acc={(tp+tn)/n:.4f} recall={tp/max(tp+fn,1):.4f} "
          f"fpr={fp/max(fp+tn,1):.4f} hit={np.mean(hits):.4f}", flush=True)
    for c, (ok, tot) in sorted(by_cat.items()):
        print(f"  {c:16s} n={tot:4d} acc={ok/tot:.3f}", flush=True)


if __name__ == "__main__":
    main()
