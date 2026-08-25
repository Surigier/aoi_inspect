"""赛题口径的完整测试集成绩单 —— 把测试做成赛题真实的样子。

此前的问题:每类只测 40缺陷 + 40正常 = 80张,而赛题是 **1000+张混合流**
(同一产品、各种缺陷类型、加上大量正常图,一张接一张进来)。80张不只是"少",
统计性质就不同——**一张图翻转就是1.25%准确率**,而我们一直在用0.01量级的差异
判断改进还是退步,那基本在噪声里。

而且砍到80张**没有任何收益**:一张图locate约150ms,650张也只要98秒,
一次fit却要20分钟。评测从来不是瓶颈。

本脚本:fit协议不变(100正常+30缺陷,30张打乱全池取、按比例涵盖所有缺陷类型),
但**测试用光该类目剩余的全部图**——缺陷与正常**混合打乱成一条流**,逐张送入,
和赛场一致。每类实际规模650~723张。

用法:PYTHONPATH=. python scripts/run_scorecard_full.py [类目...]
"""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import _read, gt_boxes, box_hit, RI, RJ, HW

CATS = ["phone_battery", "sim_card_set", "pcb", "usb", "usb_adaptor", "audiojack",
        "button_battery", "switch", "terminalblock", "transistor1", "regulator", "end_cap"]
OUT = Path("_logs/scorecard_full.json")


def prep_full(cat, n_norm=100, n_fit=30, seed=0):
    """fit与原口径完全一致;测试用光剩余全部图,并把缺陷与正常混合打乱成一条流。"""
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]
    random.Random(seed).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:n_norm]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
    random.Random(seed).shuffle(ng)
    fit_i = [load_fast(R / x["image_path"]) for x in ng[:n_fit]]
    fit_m = [_read(R / x["mask_path"], HW) for x in ng[:n_fit]]
    # 测试流:剩余全部缺陷 + 全部正常,混合打乱(赛场就是这样一张张进来的)
    stream = [(R / x["image_path"], R / x["mask_path"]) for x in ng[n_fit:]] + \
             [(R / x["image_path"], None) for x in d["test"] if x["anomaly_class"] == "OK"]
    random.Random(seed + 1).shuffle(stream)
    return normals, fit_i, fit_m, stream


def evaluate(cat):
    normals, fit_i, fit_m, stream = prep_full(cat)
    n_def = sum(1 for _, m in stream if m is not None)
    print(f"{cat}: fit 正常{len(normals)}+缺陷{len(fit_i)} | 测试流 {len(stream)}张"
          f"(缺陷{n_def} + 正常{len(stream)-n_def},已混合打乱)", flush=True)
    t0 = time.time()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"  fit完成 {time.time()-t0:.0f}s", flush=True)

    ok = 0; ious = []; hits = []; lats = []
    tp = fp = fn = tn = 0
    for ip, mp in stream:
        img = load_fast(ip)
        t1 = time.time(); o = det.locate(img); lats.append((time.time() - t1) * 1000)
        is_def_true = mp is not None
        pred = bool(o["is_defect"])
        ok += (pred == is_def_true)
        if is_def_true and pred: tp += 1
        elif is_def_true and not pred: fn += 1
        elif (not is_def_true) and pred: fp += 1
        else: tn += 1
        if not is_def_true:
            continue
        gt = _read(mp, HW)
        if not pred or o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        p = o["mask"].astype(bool); g = gt.astype(bool)
        TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
        ious.append(TP / max(TP + FP + FN, 1))
        h = box_hit(o["boxes"], gt_boxes(gt)); hits.append(h if h is not None else 0.0)

    n = len(stream)
    acc = ok / max(n, 1)
    print(f"  图级acc={acc:.3f} (TP{tp}/FN{fn}/FP{fp}/TN{tn}) | 含漏检IoU={np.mean(ious):.3f} "
          f"| 框命中@0.5={np.mean(hits):.3f} | locate中位={np.median(lats):.0f}ms "
          f"p90={np.percentile(lats,90):.0f}ms", flush=True)
    return dict(n=n, acc=acc, iou=float(np.mean(ious)), hit=float(np.mean(hits)),
                recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                lat_p90=float(np.percentile(lats, 90)))


def main(cats):
    torch.manual_seed(0)
    OUT.parent.mkdir(exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c in cats:
        if c in done:
            print(f"{c}: 已有结果,跳过(要重算加 --fresh)", flush=True); continue
        try:
            done[c] = evaluate(c)
        except Exception as e:
            print(f"{c} 失败: {type(e).__name__}: {e}", flush=True); continue
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    if done:
        k = list(done)
        tot = sum(done[c]["n"] for c in k)
        print(f"\n=== 汇总 n={len(k)}类目,共{tot}张测试图 ===", flush=True)
        for f, lab in [("acc", "图级acc"), ("iou", "含漏检IoU"), ("hit", "框命中@0.5"),
                       ("recall", "召回"), ("fpr", "误报率")]:
            v = [done[c][f] for c in k]
            print(f"  {lab:10s} 均值={np.mean(v):.3f}  最低={np.min(v):.3f}({k[int(np.argmin(v))]})", flush=True)
    print("FULL_SCORECARD OK", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--fresh" in sys.argv and OUT.exists():
        OUT.unlink()
    main(args or CATS)
