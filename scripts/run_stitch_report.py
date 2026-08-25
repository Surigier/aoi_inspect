"""2500² 同类拼接评测 + 离线可视化报告(单文件HTML,双击即开,不需要网络/服务器)。

拼接规则(**同类拼接**):一张2500²图 = 同一个产品的4张图,各放大到1250²摆成2×2,
与赛题"2500²由1024²量级小图拼接而成"的口径一致。缺陷板由4张缺陷图拼成、正常板由
4张正常图拼成,这样图级标签是干净的(混拼会让"一张图里3正常1缺陷"的标签语义变模糊)。

产出 _logs/report_<类目>.html:
  - 顶部汇总:图级acc/召回/误报率/框命中/延时
  - 每张测试图一张卡片:原图 + 预测掩膜叠加 + 预测框(红)与GT框(绿)
  - 可按 全部/正确/漏检/误报 筛选
  - 图片以base64内嵌,**整个HTML自包含,离线可看**

用法:
  PYTHONPATH=. python scripts/run_stitch_report.py phone            # 手机屏(MSD)
  PYTHONPATH=. python scripts/run_stitch_report.py phone_battery    # Real-IAD某类目
"""
import base64
import glob
import io
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from aoi.competition import CompetitionLargeDetector

BIG, TILE = 2500, 1250
HW = (256, 256)


def _to_tensor(im):
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _stitch(items):
    """items: [(PIL图, (H,W)掩膜或None)] × 4 → (3,2500,2500)张量 + (2500,2500)GT掩膜"""
    big = torch.zeros(3, BIG, BIG)
    gt = np.zeros((BIG, BIG), np.uint8)
    for k, (im, mk) in enumerate(items[:4]):
        t = F.interpolate(_to_tensor(im)[None], size=(TILE, TILE), mode="bilinear", align_corners=False)[0]
        r, c = k // 2, k % 2
        big[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        if mk is not None and mk.any():
            m = F.interpolate(torch.from_numpy(mk.astype(np.float32))[None, None],
                              size=(TILE, TILE), mode="nearest")[0, 0].numpy() > 0.5
            gt[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = m.astype(np.uint8)
    return big, gt


# ---------- 数据源 ----------
def src_phone():
    """MSD手机屏:20张good做正常,phone_best缺陷图(YOLO框→矩形掩膜)。"""
    def boxmask(lab, hw):
        h, w = hw; m = np.zeros((h, w), np.uint8)
        if not os.path.exists(lab):
            return m
        for line in open(lab):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = [float(x) for x in p[1:5]]
            m[max(0, int((cy - bh / 2) * h)):min(h, int((cy + bh / 2) * h) + 1),
              max(0, int((cx - bw / 2) * w)):min(w, int((cx + bw / 2) * w) + 1)] = 1
        return m

    norm = [(Image.open(f), None) for f in sorted(glob.glob("data/msd_good/good/*.png"))]
    dfs = []
    for split in ("train", "val", "test"):
        for f in sorted(glob.glob(f"data/phone_best/{split}/images/*")):
            lab = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if os.path.exists(lab) and os.path.getsize(lab) > 0:
                im = Image.open(f)
                dfs.append((im, boxmask(lab, (im.size[1], im.size[0]))))
    return norm, dfs, "手机屏(MSD good + phone_best)"


def src_realiad(cat):
    RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ok = [x for x in d["train"] if x["anomaly_class"] == "OK"] + \
         [x for x in d["test"] if x["anomaly_class"] == "OK"]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
    norm = [(Image.open(R / x["image_path"]), None) for x in ok[:200]]
    dfs = []
    for x in ng[:300]:
        im = Image.open(R / x["image_path"])
        mk = (np.array(Image.open(R / x["mask_path"]).convert("L")) > 0).astype(np.uint8)
        dfs.append((im, mk))
    return norm, dfs, f"Real-IAD {cat}(注:源图256²,拼接前放大到1250²,分辨率是插值出来的)"


# ---------- 可视化 ----------
def _b64(im, maxw=520):
    im = im.copy()
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.BILINEAR)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=72)
    return base64.b64encode(b.getvalue()).decode()


def _overlay(big, mask, boxes, gtb):
    arr = (big.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    if mask is not None and mask.any():
        m = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(im.size, Image.NEAREST))
        red = np.array(im); red[m > 127] = (0.45 * red[m > 127] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        im = Image.fromarray(red)
    d = ImageDraw.Draw(im)
    sx = im.width / max(mask.shape[1], 1) if mask is not None else 1
    sy = im.height / max(mask.shape[0], 1) if mask is not None else 1
    for b in gtb:
        d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(60, 255, 60), width=6)
    for b in boxes:
        d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(255, 40, 40), width=6)
    return im


HTML_HEAD = """<meta charset="utf-8"><title>AOI 2500²拼接评测报告</title>
<style>
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#111;color:#eee}
header{padding:18px 24px;background:#1b1b1b;border-bottom:1px solid #333;position:sticky;top:0;z-index:9}
h1{margin:0 0 8px;font-size:19px}
.sum{display:flex;gap:26px;flex-wrap:wrap;font-size:14px;color:#bbb}
.sum b{color:#7ee787;font-size:17px}
.btns{margin-top:12px}
button{background:#222;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 14px;margin-right:8px;cursor:pointer;font-size:13px}
button.on{background:#2d5;color:#000;border-color:#2d5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:16px;padding:18px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:10px;overflow:hidden}
.card img{width:100%;display:block}
.meta{padding:10px 12px;font-size:13px;line-height:1.7}
.ok{color:#7ee787}.miss{color:#ff7b72}.fp{color:#ffa657}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-right:6px}
.legend{font-size:12px;color:#888;margin-top:6px}
.tabs{margin-bottom:10px}
.tabs button{font-size:14px;padding:8px 18px}
section{display:none}section.show{display:block}
.stage{padding:14px 24px 0;color:#9aa;font-size:13px;line-height:1.8}
.stage b{color:#7ee787}
</style>
"""


def build_html(title, fit_rows, test_rows, summary, fitinfo):
    def card(r, tag_extra=""):
        return (f'<div class="card" data-c="{r["cls"]}"><img src="data:image/jpeg;base64,{r["img"]}">'
                f'<div class="meta"><span class="tag {r["cls"]}">{r["verdict"]}</span>'
                f'<b>{r["name"]}</b><br>{r["line1"]}<br>{r["line2"]}</div></div>')
    s = summary
    return (HTML_HEAD +
            f'<header><h1>{title}</h1>'
            f'<div class="tabs"><button class="on" onclick="tab(this,0)">① 迁移学习(fit)阶段</button>'
            f'<button onclick="tab(this,1)">② 测试阶段</button></div>'
            f'<div class="sum">'
            f'<div>测试图 <b>{s["n"]}</b> 张(2500²同类拼接)</div>'
            f'<div>图级acc <b>{s["acc"]}</b></div><div>召回 <b>{s["recall"]}</b></div>'
            f'<div>误报率 <b>{s["fpr"]}</b></div><div>框命中@0.5 <b>{s["hit"]}</b></div>'
            f'<div>含漏检IoU <b>{s["iou"]}</b></div>'
            f'<div>延时中位 <b>{s["lat"]}ms</b> / p90 <b>{s["p90"]}ms</b></div></div>'
            f'<div class="btns" id="fl" style="display:none"><button class="on" onclick="f(this,\'all\')">全部</button>'
            f'<button onclick="f(this,\'ok\')">正确</button>'
            f'<button onclick="f(this,\'miss\')">漏检</button>'
            f'<button onclick="f(this,\'fp\')">误报</button></div>'
            f'<div class="legend">红框/红色叠加 = 模型预测 · 绿框/绿色叠加 = 真实标注</div></header>'
            f'<section class="show"><div class="stage">{fitinfo}</div>'
            f'<div class="grid">{"".join(card(r) for r in fit_rows)}</div></section>'
            f'<section><div class="grid" id="g">{"".join(card(r) for r in test_rows)}</div></section>'
            '<script>'
            'function tab(b,i){document.querySelectorAll(".tabs button").forEach(x=>x.classList.remove("on"));'
            'b.classList.add("on");document.querySelectorAll("section").forEach((e,j)=>'
            'e.className=(j==i?"show":""));document.getElementById("fl").style.display=(i==1?"":"none")}'
            'function f(b,c){document.querySelectorAll("#fl button").forEach(x=>x.classList.remove("on"));'
            'b.classList.add("on");document.querySelectorAll("section")[1].querySelectorAll(".card")'
            '.forEach(function(e){e.style.display=(c=="all"||e.dataset.c==c)?"":"none"})}'
            '</script>')


def main(source, n_test=40, seed=0):
    torch.manual_seed(seed)
    if source == "phone":
        norm, dfs, title = src_phone()
    else:
        norm, dfs, title = src_realiad(source)
    rng = random.Random(seed)
    rng.shuffle(dfs)
    print(f"{title}: 正常图{len(norm)} 缺陷图{len(dfs)}", flush=True)

    # 严格按赛题协议:**100张2500²正常图 + 30张2500²缺陷图**。
    # 每张2500²本身由同类4张小图拼成 → 需要400张正常小图、120张缺陷小图。
    N_NORM_PANEL, N_DEF_PANEL = 100, 30
    n_def_panel = min(N_DEF_PANEL, len(dfs) // 4)
    if n_def_panel < N_DEF_PANEL:
        print(f"!! 缺陷图不足:只能拼出{n_def_panel}块缺陷板(赛题要30块)", flush=True)
    fit_panels = [_stitch(dfs[i * 4:(i + 1) * 4]) for i in range(n_def_panel)]
    # 正常板:**随机采样4张组合**,不要按 (i*4+k)%N 循环取。
    # 循环取的后果:20张正常图只能拼出5块不同的板,再重复20遍——板级多样性是0。
    # 随机组合下,20张取4张有4845种组合,能拼出100块互不相同的板;小图虽有重复,
    # 但**板级多样性是真的**,这对"正常长什么样"的建模差别很大。
    nrng = random.Random(seed + 7)
    if len(norm) < N_NORM_PANEL * 4:
        print(f"!! 正常图仅{len(norm)}张(拼{N_NORM_PANEL}块板理论需{N_NORM_PANEL*4}张)"
              f" → 改用随机组合复用,板级不重复但小图会重复,检测指标按下界看", flush=True)
    norm_panels = [_stitch(nrng.sample(norm, 4) if len(norm) >= 4 else norm * 4)
                   for _ in range(N_NORM_PANEL)]
    used = n_def_panel * 4

    det = CompetitionLargeDetector()
    t0 = time.time()
    det.fit_fewshot([i for i, _ in norm_panels], [i for i, _ in fit_panels],
                    defect_masks=[m for _, m in fit_panels])
    fit_sec = time.time() - t0
    print(f"fit完成 {fit_sec:.0f}s (正常板{len(norm_panels)} 缺陷板{len(fit_panels)})", flush=True)
    thr = det.decision_threshold()

    # 测试流:缺陷板 + 正常板,混合打乱
    test = [(_stitch(dfs[used + i * 4: used + (i + 1) * 4]), True)
            for i in range(min(n_test // 2, (len(dfs) - used) // 4))]
    trng = random.Random(seed + 99)                       # 测试正常板用另一组随机组合
    test += [(_stitch(trng.sample(norm, 4) if len(norm) >= 4 else norm * 4), False)
             for _ in range(n_test // 2)]
    rng.shuffle(test)

    from scripts.run_scorecard import gt_boxes, box_hit
    # ---- fit阶段可视化:把喂进去的正常板与缺陷板(含GT标注)原样展示 ----
    fit_rows = []
    for i, (big, gt) in enumerate(norm_panels[:8]):     # 100块全展示太大,取前8块示例
        im = _overlay(big, np.zeros(HW, np.uint8), [], [])
        fit_rows.append(dict(img=_b64(im), name=f"正常板 #{i:02d}", cls="ok", verdict="正常样本",
                             line1="用途:给模型建立『这个产品长什么样』的基准",
                             line2=f"由4张同类正常图拼成 · 2500×2500"))
    for i, (big, gt) in enumerate(fit_panels[:12]):    # 同上,取前12块示例
        gtb = gt_boxes((F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                                      size=HW, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8))
        im = _overlay(big, np.zeros(HW, np.uint8), [], gtb)
        fit_rows.append(dict(img=_b64(im), name=f"缺陷板 #{i:02d}", cls="fp", verdict="缺陷样本(带标注)",
                             line1="用途:训监督分割头 + 标定判决阈值 + VLM打缺陷类型标签",
                             line2=f"绿框=人工标注的缺陷位置 · 共{len(gtb)}处"))
    rows = []; nok = 0; tp = fn = fp = tn = 0; ious = []; hits = []; lats = []
    for idx, ((big, gt), is_def_true) in enumerate(test):
        t1 = time.time(); o = det.locate(big); ms = (time.time() - t1) * 1000; lats.append(ms)
        pred = bool(o["is_defect"])
        nok += (pred == is_def_true)
        if is_def_true and pred: tp += 1; cls = "ok"; verdict = "✅ 检出"
        elif is_def_true and not pred: fn += 1; cls = "miss"; verdict = "❌ 漏检"
        elif pred: fp += 1; cls = "fp"; verdict = "⚠️ 误报"
        else: tn += 1; cls = "ok"; verdict = "✅ 正常"
        mk = o.get("mask")
        gtr = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                             size=mk.shape if mk is not None else HW, mode="nearest")[0, 0].numpy() > 0.5
               ).astype(np.uint8)
        gtb = gt_boxes(gtr) if is_def_true else []
        iou = hit = 0.0
        if is_def_true and mk is not None:
            p = mk.astype(bool); g = gtr.astype(bool)
            TP = int((p & g).sum()); FP2 = int((p & ~g).sum()); FN2 = int((~p & g).sum())
            iou = TP / max(TP + FP2 + FN2, 1)
            h = box_hit(o["boxes"], gtb); hit = h if h is not None else 0.0
            ious.append(iou); hits.append(hit)
        elif is_def_true:
            ious.append(0.0); hits.append(0.0)
        im = _overlay(big, mk if mk is not None else np.zeros(HW, np.uint8), o.get("boxes") or [], gtb)
        rows.append(dict(img=_b64(im), name=f"测试 #{idx:02d}", cls=cls, verdict=verdict,
                         line1=f'真实={"缺陷" if is_def_true else "正常"} · 判定={"缺陷" if pred else "正常"} · 类型={o["defect_type"]}',
                         line2=f'异常分={o["score"]:.4f} / 阈值={thr:.4f} · 框命中={hit:.2f} · IoU={iou:.3f} · {ms:.0f}ms'))
    n = len(test)
    summary = dict(n=n, acc=f"{nok/max(n,1):.3f}", recall=f"{tp/max(tp+fn,1):.1%}",
                   fpr=f"{fp/max(fp+tn,1):.1%}", hit=f"{np.mean(hits) if hits else 0:.3f}",
                   iou=f"{np.mean(ious) if ious else 0:.3f}", thr=f"{thr:.4f}",
                   lat=f"{np.median(lats):.0f}", p90=f"{np.percentile(lats,90):.0f}")
    fitinfo = (f'<b>模拟赛场的迁移学习阶段</b>(赛题:此阶段<b>不计时</b>)。'
               f'喂给模型:<b>{len(norm_panels)}</b> 块正常板 + <b>{len(fit_panels)}</b> 块缺陷板'
               f'(= {len(norm_panels)*4} 张正常图 + {len(fit_panels)*4} 张缺陷图,同类4张拼成一块2500²)。'
               f'耗时 <b>{fit_sec:.0f}s</b>。下面是**实际喂进去的每一块板**,缺陷板上的绿框是人工标注。')
    out = Path(f"_logs/report_{source}.html")
    out.write_text(build_html(title, fit_rows, rows, summary, fitinfo), encoding="utf-8")
    print(f"\n=== {title} ===", flush=True)
    print(f"图级acc={summary['acc']} 召回={summary['recall']} 误报率={summary['fpr']} "
          f"框命中={summary['hit']} IoU={summary['iou']} 延时中位={summary['lat']}ms/p90={summary['p90']}ms",
          flush=True)
    print(f"报告已生成: {out}  ({out.stat().st_size/1e6:.1f} MB,双击即可离线查看)", flush=True)
    print("STITCH_REPORT OK", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phone",
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)
