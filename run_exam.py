"""模拟考:**一次 fit、一条混合测试流**,完全按赛题走。

和此前"每个类目各跑一遍"的区别 —— 赛场只有**一次**迁移学习,不是12次:
  fit  : 100张2500²正常图 + 30张2500²缺陷图(从选定的几个手机部件类目里凑)
  测试 : 1000张2500²图,缺陷与正常**混合打乱**成一条流,逐张送入

拼接规则(**同类拼接**):每张2500² = 同一产品的4张图各放大到1250²摆2×2,
对应赛题"2500²由1024²量级小图拼接而成"。板内同类,板间跨类——这正是
"手机屏幕/电池/中框"多种部件混在一条测试流里的样子。

选定类目(Real-IAD里三个**字面意义上的手机部件**):
  phone_battery(手机电池) / sim_card_set(SIM卡座) / pcb(主板)

产出:
  _logs/exam_report.html  单文件离线报告(fit阶段 + 测试阶段可视化)
  控制台:图级acc / 召回 / 误报率 / 框命中@0.5 / 含漏检IoU / 延时

用法:PYTHONPATH=. python scripts/run_exam.py [测试张数=1000]
"""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from scripts.run_stitch_report import (_stitch, _overlay, _b64, build_html, HW,
                                       BIG, TILE)
from scripts.run_scorecard import gt_boxes, box_hit

import os as _os
# 默认三个手机部件混跑;EXAM_CATS 可指定单类目,用来分离"混类"与"拼接稀释"两个因素
CATS = _os.environ.get("EXAM_CATS", "phone_battery,sim_card_set,pcb").split(",")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
N_FIT_NORM, N_FIT_DEF = 100, 30                       # 赛题协议
DEF_RATIO = 0.3                                       # 测试流里含缺陷板的比例(产线现实:缺陷是少数)


def load_pool():
    """按类目收集 (正常图列表, 缺陷图+掩膜列表)。"""
    pool = {}
    for c in CATS:
        d = json.load(open(RJ / f"{c}.json")); R = RI / c
        ok = [R / x["image_path"] for x in d["train"] if x["anomaly_class"] == "OK"] + \
             [R / x["image_path"] for x in d["test"] if x["anomaly_class"] == "OK"]
        ng = [(R / x["image_path"], R / x["mask_path"])
              for x in d["test"] if x["anomaly_class"] != "OK"]
        pool[c] = (ok, ng)
        print(f"  {c}: 正常{len(ok)}张 缺陷{len(ng)}张", flush=True)
    return pool


# 固定顺序取图,不随机——每类维护一个游标,连续取4张拼一块板。
# 好处:完全可复现,同一份数据每次跑出的板一模一样,便于对比不同改动。
_CUR = {}


def _take(pool, cat, kind, k=4):
    key = (cat, kind)
    lst = pool[cat][0 if kind == "ok" else 1]
    i = _CUR.get(key, 0)
    out = [lst[(i + j) % len(lst)] for j in range(k)]
    _CUR[key] = (i + k) % len(lst)
    return out


def _norm_panel(pool, cat, rng=None):
    return _stitch([(Image.open(p), None) for p in _take(pool, cat, "ok")])


def _def_panel(pool, cat, rng=None):
    items = []
    for ip, mp in _take(pool, cat, "ng"):
        mk = (np.array(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        items.append((Image.open(ip), mk))
    return _stitch(items)


def main(n_test=1000, seed=0, per_mode=False, seg_gate=False):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    print("=== 模拟考:一次fit + 一条混合测试流(严格按赛题)===", flush=True)
    pool = load_pool()

    # ---- fit:100正常板 + 30缺陷板,跨三个类目均摊 ----
    fit_norm, fit_def = [], []
    for i in range(N_FIT_NORM):
        fit_norm.append(_norm_panel(pool, CATS[i % len(CATS)], rng))
    for i in range(N_FIT_DEF):
        fit_def.append((_def_panel(pool, CATS[i % len(CATS)], rng), CATS[i % len(CATS)]))
    print(f"fit输入: 正常板{len(fit_norm)} + 缺陷板{len(fit_def)}"
          f"(跨{len(CATS)}类均摊,每块2500²=同类4张拼)", flush=True)

    det = CompetitionLargeDetector(per_mode_gate=per_mode, seg_gate=seg_gate)
    print(f"per_mode_gate={per_mode}  seg_gate={seg_gate}", flush=True)
    t0 = time.time()
    det.fit_fewshot([b for b, _ in fit_norm], [b for (b, _), _ in fit_def],
                    defect_masks=[m for (_, m), _ in fit_def])
    fit_sec = time.time() - t0
    thr = det.decision_threshold()
    print(f"fit完成 {fit_sec:.0f}s  阈值={thr:.4f}", flush=True)

    # ---- 测试流:n_test 张,缺陷占 DEF_RATIO,混合打乱 ----
    # 测试流**必须惰性拼接**:每块2500²×3通道float32 = 75MB,1000块预先拼好就是75GB,
    # 直接把系统内存打爆(实测被内核OOM杀掉,anon-rss 30GB / 总内存31GB)。
    # 这里只先决定"第几张是什么类目、是缺陷还是正常",真正的图在循环里现拼现用完就丢。
    n_def = int(n_test * DEF_RATIO)
    plan = [(CATS[i % len(CATS)], True) for i in range(n_def)] + \
           [(CATS[i % len(CATS)], False) for i in range(n_test - n_def)]
    rng.shuffle(plan)            # 只打乱顺序(模拟混合流),取图本身是固定的
    print(f"测试流: {len(plan)}张(缺陷{n_def} + 正常{n_test-n_def},已混合打乱,惰性拼接)", flush=True)

    nok = tp = fn = fp = tn = 0
    ious, hits, lats, rows = [], [], [], []
    scores, labels = [], []                      # 供阈值扫描复用,免去重新fit
    for idx, (cat, is_def) in enumerate(plan):
        big, gt = _def_panel(pool, cat) if is_def else _norm_panel(pool, cat)
        t1 = time.time(); o = det.locate(big); ms = (time.time() - t1) * 1000; lats.append(ms)
        scores.append(float(o["score"])); labels.append(bool(is_def))
        pred = bool(o["is_defect"]); nok += (pred == is_def)
        if is_def and pred: tp += 1; cls, verdict = "ok", "✅ 检出"
        elif is_def: fn += 1; cls, verdict = "miss", "❌ 漏检"
        elif pred: fp += 1; cls, verdict = "fp", "⚠️ 误报"
        else: tn += 1; cls, verdict = "ok", "✅ 正常"
        mk = o.get("mask")
        iou = hit = 0.0; gtb = []
        if is_def:
            gtr = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                                 size=mk.shape if mk is not None else HW,
                                 mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
            gtb = gt_boxes(gtr)
            if mk is not None:
                p_ = mk.astype(bool); g_ = gtr.astype(bool)
                TP = int((p_ & g_).sum()); FP2 = int((p_ & ~g_).sum()); FN2 = int((~p_ & g_).sum())
                iou = TP / max(TP + FP2 + FN2, 1)
                h = box_hit(o["boxes"], gtb); hit = h if h is not None else 0.0
            ious.append(iou); hits.append(hit)
        if len(rows) < 120:                                  # 报告只留前120张,否则HTML过大
            im = _overlay(big, mk if mk is not None else np.zeros(HW, np.uint8),
                          o.get("boxes") or [], gtb)
            rows.append(dict(img=_b64(im), name=f"测试 #{idx:04d} · {cat}", cls=cls, verdict=verdict,
                             line1=f'真实={"缺陷" if is_def else "正常"} · 判定={"缺陷" if pred else "正常"} · 类型={o["defect_type"]}',
                             line2=f'异常分={o["score"]:.4f} / 阈值={thr:.4f} · 框命中={hit:.2f} · IoU={iou:.3f} · {ms:.0f}ms'))
        del big, gt                                   # 用完立刻释放,峰值内存只占一张板
        if (idx + 1) % 100 == 0:
            print(f"  已测 {idx+1}/{len(plan)}  当前acc={nok/(idx+1):.3f}", flush=True)

    # ---- 阈值扫描:同一次fit的分数复用,几秒钟就能看清"阈值到底值多少分" ----
    # 关键:阈值改动不需要重新fit(1小时),分数是固定的,只是切的位置不同。
    sc = np.array(scores); lb = np.array(labels, dtype=bool)
    cands = np.unique(np.concatenate([sc, [thr]]))
    accs = np.array([(((sc >= t) == lb).mean()) for t in cands])
    bi = int(np.argmax(accs))
    cur_acc = ((sc >= thr) == lb).mean()
    print(f"\n=== 阈值扫描(同一次fit,{len(cands)}个候选)===", flush=True)
    print(f"  当前标定阈值 {thr:.6g} → acc={cur_acc:.3f}  "
          f"(召回{((sc>=thr)&lb).sum()/max(lb.sum(),1):.1%} 误报{((sc>=thr)&~lb).sum()/max((~lb).sum(),1):.1%})",
          flush=True)
    print(f"  最优阈值     {cands[bi]:.6g} → acc={accs[bi]:.3f}  "
          f"(召回{((sc>=cands[bi])&lb).sum()/max(lb.sum(),1):.1%} "
          f"误报{((sc>=cands[bi])&~lb).sum()/max((~lb).sum(),1):.1%})", flush=True)
    print(f"  **阈值标定的损失 = {accs[bi]-cur_acc:+.3f} acc**  ← 这是纯靠改阈值就能拿回来的分",
          flush=True)
    print(f"  分数分布:正常 中位={np.median(sc[~lb]):.6g} p99={np.percentile(sc[~lb],99):.6g} | "
          f"缺陷 中位={np.median(sc[lb]):.6g} p1={np.percentile(sc[lb],1):.6g}", flush=True)
    print(f"  两类重叠度:{((sc[lb]<=sc[~lb].max())).mean():.1%} 的缺陷分落在正常图分数区间内",
          flush=True)

    n = len(plan)
    summary = dict(n=n, acc=f"{nok/n:.3f}", recall=f"{tp/max(tp+fn,1):.1%}",
                   fpr=f"{fp/max(fp+tn,1):.1%}", hit=f"{np.mean(hits):.3f}",
                   iou=f"{np.mean(ious):.3f}", thr=f"{thr:.4f}",
                   lat=f"{np.median(lats):.0f}", p90=f"{np.percentile(lats,90):.0f}")
    fit_rows = []
    for i, (big, _) in enumerate(fit_norm[:8]):
        fit_rows.append(dict(img=_b64(_overlay(big, np.zeros(HW, np.uint8), [], [])),
                             name=f"正常板 #{i:02d}", cls="ok", verdict="正常样本",
                             line1="用途:建立『这个产品长什么样』的基准",
                             line2="同类4张拼成2500² · 共100块"))
    for i, ((big, gt), cat) in enumerate(fit_def[:12]):
        gtr = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                             size=HW, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        gtb = gt_boxes(gtr)
        fit_rows.append(dict(img=_b64(_overlay(big, np.zeros(HW, np.uint8), [], gtb)),
                             name=f"缺陷板 #{i:02d} · {cat}", cls="fp", verdict="缺陷样本(带标注)",
                             line1="用途:训监督分割头 + 标定阈值 + VLM打类型标签",
                             line2=f"绿框=人工标注缺陷位置,共{len(gtb)}处 · 共30块"))
    fitinfo = (f'<b>迁移学习阶段</b>(赛题此阶段<b>不计时</b>)。喂入 <b>100</b> 块正常板 + '
               f'<b>30</b> 块缺陷板,跨 <b>{"/".join(CATS)}</b> 三个手机部件类目均摊,'
               f'每块 2500²(同类4张拼)。耗时 <b>{fit_sec:.0f}s</b>,标定阈值 <b>{thr:.4f}</b>。'
               f'下面是示例板,缺陷板绿框为人工标注。')
    tag = ("_permode" if per_mode else "") + ("_seggate" if seg_gate else "")
    out = Path(f"_logs/exam_report{tag}.html")
    out.write_text(build_html("AOI 模拟考 · 2500²混合流(phone_battery/sim_card_set/pcb)",
                              fit_rows, rows, summary, fitinfo), encoding="utf-8")
    print(f"\n=== 模拟考结果({n}张混合流)===", flush=True)
    print(f"图级acc={summary['acc']} (TP{tp}/FN{fn}/FP{fp}/TN{tn})", flush=True)
    print(f"召回={summary['recall']}  误报率={summary['fpr']}", flush=True)
    print(f"框命中@0.5={summary['hit']}  含漏检IoU={summary['iou']}", flush=True)
    print(f"延时 中位={summary['lat']}ms  p90={summary['p90']}ms  (预算200ms)", flush=True)
    print(f"报告: {out} ({out.stat().st_size/1e6:.1f} MB,双击离线可看)", flush=True)
    print("EXAM OK", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 1000, per_mode="--per-mode" in sys.argv,
         seg_gate="--seg-gate" in sys.argv)
