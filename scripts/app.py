"""AOI 实时在线 AI 质检 · 交付演示台(Web端)

按赛题的真实工作流组织,工作人员**勾选文件**即可完成两个阶段:
  ① 迁移学习(fit) —— 选正常图/缺陷图/掩膜 → 现场迁移(赛题规定此阶段不计时)
  ② 在线检测(test)—— 选待检图片 → 输出判决/类型/检测框/延时,并可回溯检测逻辑
  ③ 交付汇总      —— 方案、成绩、已验证与已判负的路线

用法:PYTHONPATH=. python scripts/app.py [--port 7860] [--root demo_data]
"""
import argparse
import csv
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parent.parent / "models" / "hf_cache"))

from aoi.competition import CompetitionLargeDetector          # noqa: E402
from aoi.active_learning import ActiveLearningLoop             # noqa: E402

ROOT = Path("demo_data")
STATE = {"det": None, "loop": None, "product": None, "fit_sec": 0}


# ---------------- 基础 ----------------
def _load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _mask(p, hw=(256, 256)):
    return (np.array(Image.open(p).convert("L").resize(hw[::-1], Image.NEAREST)) > 0).astype(np.uint8)


def products():
    return sorted([d.name for d in ROOT.iterdir() if d.is_dir()]) if ROOT.exists() else []


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def files_of(product, sub):
    d = ROOT / product / sub
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def _seg_fp(det):
    """监督分割头的权重指纹——用来向操作员证明反馈**真的改了模型参数**。"""
    import hashlib
    if det.seg_head.head is None:
        return "无头"
    b = b"".join(p.detach().cpu().numpy().tobytes() for p in det.seg_head.head.parameters())
    return hashlib.md5(b).hexdigest()[:12]


def render(img_t, o=None, gt_boxes=None, ms=None):
    """原图 + 预测掩膜叠加(红) + 预测框(红) + 人工标注框(绿)"""
    arr = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    if o is not None and o.get("mask") is not None and o["mask"].any():
        m = np.array(Image.fromarray((o["mask"] * 255).astype(np.uint8)).resize(im.size, Image.NEAREST))
        a = np.array(im)
        a[m > 127] = (0.45 * a[m > 127] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    w = max(3, im.width // 220)
    if o is not None and o.get("mask") is not None:
        sx, sy = im.width / o["mask"].shape[1], im.height / o["mask"].shape[0]
        for b in (o.get("boxes") or []):
            d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(255, 40, 40), width=w)
    if gt_boxes is not None:
        for b in gt_boxes:
            d.rectangle(list(b), outline=(60, 255, 60), width=w)
    if o is not None:
        tag = ("🔴 缺陷:" + o["defect_type"]) if o["is_defect"] else "🟢 正常"
        d.rectangle([0, 0, im.width, 44], fill=(0, 0, 0))
        d.text((10, 6), tag, fill=(255, 255, 0))
        d.text((10, 25), f"异常分{o['score']:.3f} · 框{len(o.get('boxes') or [])}个"
                         + (f" · {ms:.0f}ms" if ms else ""), fill=(200, 200, 200))
    return im


# ---------------- ① 迁移学习 ----------------
def preview_fit(product, n_show=6):
    if not product:
        return [], [], "请先选择产品"
    ns = files_of(product, "normal")
    ds = files_of(product, "defect")
    gn = [str(ROOT / product / "normal" / f) for f in ns[:n_show]]
    gd = []
    for f in ds[:n_show]:
        ip = ROOT / product / "defect" / f
        mp = ROOT / product / "mask" / f
        img = _load(ip)
        gb = []
        if mp.exists():
            import cv2
            m = (np.array(Image.open(mp).convert("L")) > 0).astype(np.uint8)
            nn, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            gb = [(x, y, x + w, y + h) for x, y, w, h, a in
                  (st[i] for i in range(1, nn)) if a >= 16]
        gd.append(render(img, None, gt_boxes=gb))
    info = (f"### {product}\n"
            f"- 正常图 **{len(ns)}** 张(建立『这个产品长什么样』的基准)\n"
            f"- 缺陷图 **{len(ds)}** 张 + 人工标注掩膜(训监督分割头、标定判决阈值、"
            f"VLM打缺陷类型标签)\n\n绿框 = 人工标注的缺陷位置。\n\n"
            f"点下方按钮开始迁移学习(**赛题规定此阶段不计时**,约 20 分钟)。")
    return gn, gd, info


def do_fit(product, n_norm, n_def, progress=None):
    if not product:
        return "请先选择产品"
    ns = files_of(product, "normal")[:int(n_norm)]
    ds = files_of(product, "defect")[:int(n_def)]
    if not ns or not ds:
        return f"### ⚠️ 目录不完整:normal {len(ns)} 张 / defect {len(ds)} 张,两者都不能为空"
    missing = [f for f in ds if not (ROOT / product / "mask" / f).exists()]
    if missing:
        return (f"### ⚠️ {len(missing)} 张缺陷图缺少同名掩膜(mask/ 目录):"
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    imgs = [_load(ROOT / product / "normal" / f) for f in ns]
    dimg = [_load(ROOT / product / "defect" / f) for f in ds]
    dmsk = [_mask(ROOT / product / "mask" / f) for f in ds]
    det = CompetitionLargeDetector()
    t0 = time.time()
    # 用 ActiveLearningLoop 承载:它维护样本集,操作员反馈时可增量重拟合
    loop = ActiveLearningLoop(det, imgs, dimg, defect_masks=dmsk)
    sec = time.time() - t0
    STATE.update(det=det, loop=loop, product=product, fit_sec=sec)
    th = getattr(det.type_head, "ready", False)
    return (f"### ✅ 迁移学习完成\n"
            f"- 产品:**{product}** · 正常 {len(ns)} 张 + 缺陷 {len(ds)} 张\n"
            f"- 耗时 **{sec:.0f}s**(赛题此阶段不计时)\n"
            f"- 判决阈值:`{det.decision_threshold():.4f}`\n"
            f"- 监督分割头:{'已训(用掩膜)' if det.seg_head.head is not None else '未训'}\n"
            f"- 缺陷类型头:{'已就绪' if th else '降级到规则模式'}\n"
            f"- DINOv2 图级门:{'启用' if det._dino is not None else '未启用'}\n"
            f"- 延时自适应裁剪:`{getattr(det,'lat_trimmed',[])}`\n\n"
            f"→ 切到 **② 在线检测** 页签")


# ---------------- ② 在线检测 ----------------
def _answers(product):
    p = ROOT / product / "answer.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {r[0]: r[1] for r in list(csv.reader(f))[1:]}


def run_test(files, uploads=None, progress=None):
    det, product = STATE["det"], STATE["product"]
    if det is None:
        return [], None, "### ⚠️ 请先在 ① 完成迁移学习"
    todo = [(f, ROOT / product / "test" / f) for f in (files or [])]
    # 自选文件:操作员上传任意图片(现场抓拍等),无对照答案,只出判定不计准确率
    for u in (uploads or []):
        up = Path(getattr(u, "name", u))
        if up.suffix.lower() in IMG_EXTS:
            todo.append((up.name, up))
    if not todo:
        return [], None, "### 请勾选待检文件或上传图片"
    ans = _answers(product)
    gal, rows = [], []
    tp = fn = fp = tn = 0
    lats = []
    for f, path in todo:
        img = _load(path)
        t0 = time.time(); o = det.locate(img); ms = (time.time() - t0) * 1000
        lats.append(ms)
        truth = ans.get(f, "?")
        pred = "缺陷" if o["is_defect"] else "正常"
        if truth == "缺陷" and o["is_defect"]: tp += 1; mark = "✅检出"
        elif truth == "缺陷": fn += 1; mark = "❌漏检"
        elif truth == "正常" and o["is_defect"]: fp += 1; mark = "⚠️误报"
        elif truth == "正常": tn += 1; mark = "✅正常"
        else: mark = "—"
        gal.append((render(img, o, ms=ms), f"{f} {mark}"))
        rows.append([f, truth, pred, o["defect_type"], f"{o['score']:.3f}",
                     len(o.get("boxes") or []), f"{ms:.0f}", mark])
    n = len(todo)
    n_ans = tp + fn + fp + tn          # 只有带对照答案的图才计入准确率,上传的自选图不稀释
    md = (f"### 检测完成 · {n} 张(其中 {n_ans} 张有对照答案)\n"
          f"| | 值 |\n|---|---|\n"
          f"| 检出(TP) | {tp} |\n| 漏检(FN) | {fn} |\n"
          f"| 误报(FP) | {fp} |\n| 正常正确(TN) | {tn} |\n"
          f"| **图级准确率** | **{(tp+tn)/max(n_ans,1):.1%}** |\n"
          f"| 召回 | {tp/max(tp+fn,1):.1%} |\n"
          f"| 误报率 | {fp/max(fp+tn,1):.1%} |\n"
          f"| 延时 中位/p90 | {np.median(lats):.0f} / {np.percentile(lats,90):.0f} ms(预算200ms) |\n")
    return gal, rows, md


def explain_one(fname):
    det, product = STATE["det"], STATE["product"]
    if det is None or not fname:
        return "### 请先完成迁移学习并选择一张图"
    o = det.explain(_load(ROOT / product / "test" / fname))
    import json
    return "### 检测逻辑回溯(赛题要求)\n```json\n" + json.dumps(o, ensure_ascii=False,
                                                          indent=2, default=str) + "\n```"


SUMMARY = """
## 交付汇总

### 方案
`locate()` = **EfficientAD(双学生)+ DINOv2 双判据图级检测** → 判正常立即早退
→ **WRN 浅层(1,2)@512 + 双头联合训练监督分割** → **SAM 受控精化**
→ **VLM 蒸馏的缺陷类型头**(fit 期标注、推理零 API)

### 成绩(5类目混合迁移 + 1300张混合流,三种子)
| 指标 | 值 |
|---|---|
| 图级准确率 | **0.887 ± 0.023** |
| 召回 / 误报率 | 90.7% / 12.2% |
| 框命中@0.5 | 0.373 ± 0.020 |
| 含漏检 IoU | 0.357 ± 0.016 |
| 缺陷类型归属 | 40.9% |
| 单图延时 中位 | **80 ms**(预算 200ms) |

**隐藏域(真实手机屏 1000 张缺陷图):框命中 0.697 ± 0.073,纯定位 0.808**

### 检测时间(竞赛得分 30%)
2500² 真实输入尺寸:中位 **110ms** / p90 **129ms**。
实测**核心频率砍 33% 延时几乎不变** → 不受 GPU 算力约束,RTX 2060 上守住 200ms 无悬念。

### 方法论:零回退纪律
候选改动须在真实留出数据上 `median(Δ)≥0.005 且 min(Δ)≥-0.01` 才进生产,验负立即回退并留档。
**17 个候选仅 1 个通过**(seg_head 双头联合训练)。判负清单与四次台架失误全部留档。

### 未达成(如实标注)
CPU < 2s:实测 4.4s,瓶颈单点在 EfficientAD 教师 @1152 整图卷积(占 92%)。
三条补救路径均已评估,GPU 主线已达标前提下不投入。
"""


def build():
    import gradio as gr
    ps = products()
    with gr.Blocks(title="AOI 实时在线 AI 质检") as app:
        gr.Markdown("# 🔍 可自学习的 AOI 实时在线 AI 质检")
        with gr.Tab("① 迁移学习(不计时)"):
            with gr.Row():
                root_tb = gr.Textbox(str(ROOT), label="数据根目录(可自选:下含 <产品>/normal|defect|mask|test)",
                                     scale=3)
                btn_root = gr.Button("应用目录", scale=1)
            with gr.Row():
                prod = gr.Dropdown(ps, value=(ps[0] if ps else None), label="选择产品")
                nn = gr.Number(100, label="正常图张数", precision=0)
                nd = gr.Number(30, label="缺陷图张数", precision=0)
            info = gr.Markdown()
            with gr.Row():
                g1 = gr.Gallery(label="正常样本(建立基准)", columns=6, height=200)
                g2 = gr.Gallery(label="缺陷样本 + 人工标注(绿框)", columns=6, height=200)
            btn_fit = gr.Button("开始迁移学习", variant="primary")
            fit_out = gr.Markdown()
            def _set_root(path):
                global ROOT
                d = Path(path).expanduser()
                if not d.is_dir():
                    return gr.update(), f"### ⚠️ 目录不存在:{d}"
                ROOT = d
                new_ps = products()
                return gr.update(choices=new_ps, value=(new_ps[0] if new_ps else None)), \
                       f"已切换到 `{d}`,找到 {len(new_ps)} 个产品目录"
            btn_root.click(_set_root, root_tb, [prod, info])
            prod.change(preview_fit, prod, [g1, g2, info])
            app.load(preview_fit, prod, [g1, g2, info])
            btn_fit.click(do_fit, [prod, nn, nd], fit_out)
        with gr.Tab("② 在线检测(计时)"):
            tf = gr.CheckboxGroup([], label="勾选待检文件(来自 test/ 混合流)")
            up = gr.File(label="或上传自选图片(现场抓拍等,可多张;无对照答案,不计入准确率)",
                         file_count="multiple", file_types=["image"])
            with gr.Row():
                btn_all = gr.Button("全选"); btn_run = gr.Button("开始检测", variant="primary")
            res_md = gr.Markdown()
            gal = gr.Gallery(label="检测结果(红=预测缺陷区域/预测框)", columns=4, height=520)
            tbl = gr.Dataframe(headers=["文件", "真实", "判定", "缺陷类型", "异常分", "框数", "ms", "结果"],
                               label="逐图明细")
            with gr.Row():
                one = gr.Dropdown([], label="回溯某张图的检测逻辑")
                btn_ex = gr.Button("回溯")
            ex_md = gr.Markdown()

            def _refresh(p):
                fs = files_of(p, "test")
                return gr.update(choices=fs, value=[]), gr.update(choices=fs)
            prod.change(_refresh, prod, [tf, one])
            app.load(_refresh, prod, [tf, one])
            btn_all.click(lambda p: gr.update(value=files_of(p, "test")), prod, tf)
            btn_run.click(run_test, [tf, up], [gal, tbl, res_md])
            btn_ex.click(explain_one, one, ex_md)
        with gr.Tab("③ 操作员反馈(自学习)"):
            gr.Markdown(
                "赛题要求:**当系统误检或漏检时,操作员可提供实时反馈,系统应能回溯检测逻辑,"
                "动态调整模型参数**。\n\n"
                "在 ② 里看到判错的图 → 在这里选中它、标出真实情况 → 提交反馈。系统会:\n"
                "1. **用 VLM 当场说出这是什么缺陷**(不是 z 分,是人话)\n"
                "2. **把样本并入训练集并重训监督分割头**(真正改模型参数,不只是调阈值)\n"
                "3. 重标判决阈值 / DINO门 / 像素阈值 / 缺陷类型质心\n"
                "4. 反馈样本作为**硬约束**参与阈值标定;若救回它要付出超过10%的误报"
                "(留出实测这种情况付+17.2pp误报换0召回),系统会**如实告知救不回**,"
                "而不是牺牲整条产线\n\n"
                "> 漏检反馈跳过 EAD 学生重训——学生只在正常图上训,新增的是缺陷图,"
                "重训纯属浪费。实测单轮 **1193s → 251s**,这是『实时』能成立的关键。")
            with gr.Row():
                fb_file = gr.Dropdown([], label="选择判错的图(来自 test/)")
                fb_up = gr.File(label="或上传判错的自选图片", file_count="single", file_types=["image"])
                fb_truth = gr.Radio(["缺陷(漏检了)", "正常(误报了)"], value="缺陷(漏检了)",
                                    label="操作员判定的真实情况")
            fb_btn = gr.Button("提交反馈 → 自学习", variant="primary")
            fb_img = gr.Image(label="反馈样本")
            fb_md = gr.Markdown()

            def _do_fb(fname, upfile, truth):
                loop, product = STATE["loop"], STATE["product"]
                if loop is None:
                    return None, "### ⚠️ 请先在 ① 完成迁移学习"
                if upfile is not None:                        # 上传的自选图优先
                    img = _load(getattr(upfile, "name", upfile))
                elif fname:
                    img = _load(ROOT / product / "test" / fname)
                else:
                    return None, "### 请选择一张图或上传图片"
                is_def = truth.startswith("缺陷")
                det = STATE["det"]
                before = (float(det.decision_threshold()), _seg_fp(det))
                mk = None
                if is_def:                       # 漏检反馈:用当前预测掩膜当标注(现场没有GT)
                    o = det.locate(img)
                    mk = o.get("mask")
                    if mk is None or not mk.any():
                        amap = det.segment(img)
                        mk = (amap >= np.percentile(amap, 99.5)).astype(np.uint8)
                t0 = time.time()
                n_n, n_d = loop.feedback(img, is_defect=is_def, mask=mk)
                sec = time.time() - t0
                diag = getattr(loop, "last_diagnosis", None)
                after = (float(det.decision_threshold()), _seg_fp(det))
                o2 = det.locate(img)
                md = [f"### ✅ 反馈已生效({sec:.0f}s)",
                      f"- 样本库:正常 {n_n} / 缺陷 {n_d}"]
                if diag:
                    md.append(f"- **VLM 即时诊断**:{diag.get('现象')} → 判为 **{diag.get('类型')}**")
                else:
                    md.append("- VLM 诊断不可用(无外网/无key)→ 仅记录样本,不影响自学习")
                md += ["", "#### 模型参数变化(证明是真的在改模型,不是只调阈值)",
                       "| 项 | 反馈前 | 反馈后 | |", "|---|---|---|---|",
                       f"| 图级判决阈值 | {before[0]:.4f} | {after[0]:.4f} | "
                       f"{'✅变了' if before[0]!=after[0] else '未变'} |",
                       f"| 分割头权重指纹 | `{before[1]}` | `{after[1]}` | "
                       f"{'✅重训了' if before[1]!=after[1] else '未变'} |",
                       "", f"#### 该样本反馈后的判定:"
                           f"{'🔴 缺陷' if o2['is_defect'] else '🟢 正常'} "
                           f"(融合分 {det.frame_score(img):.4f} / 阈值 {det.decision_threshold():.4f})"]
                # **救不回就说救不回**:硬约束受"fit正常图误报率≤10%"兜底,压不下去时
                # 不能让界面显示得像修好了——操作员据此才知道该补样本还是换角度拍。
                for u in (getattr(det, "_fb_unsat", None) or []):
                    md.append(f"\n> ⚠️ **该反馈无法完全满足**:{u}。系统已保住误报率上限,"
                              f"未强行压低阈值。建议补几张同类缺陷样本后重新迁移。")
                return render(img, o2), "\n".join(md)

            fb_btn.click(_do_fb, [fb_file, fb_up, fb_truth], [fb_img, fb_md])
            prod.change(lambda p: gr.update(choices=files_of(p, "test")), prod, fb_file)
            app.load(lambda p: gr.update(choices=files_of(p, "test")), prod, fb_file)

        with gr.Tab("④ 交付汇总"):
            gr.Markdown(SUMMARY)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--root", default="demo_data")
    a = ap.parse_args()
    ROOT = Path(a.root)
    build().launch(server_name="0.0.0.0", server_port=a.port, share=False)
