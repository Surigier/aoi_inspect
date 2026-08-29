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

import cv2
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
    return _fit_core(imgs, dimg, dmsk, product)


def do_fit_files(nfiles, dfiles, mfiles):
    """自选文件版迁移学习:三组文件由系统文件框选(不依赖目录结构)。
    掩膜按**文件名(不含扩展名)**与缺陷图一一配对。"""
    def _paths(fs):
        return [Path(getattr(u, "name", u)) for u in (fs or [])
                if Path(getattr(u, "name", u)).suffix.lower() in IMG_EXTS]
    np_, dp_, mp_ = _paths(nfiles), _paths(dfiles), _paths(mfiles)
    if not np_ or not dp_:
        return f"### ⚠️ 正常图 {len(np_)} 张 / 缺陷图 {len(dp_)} 张,两组都不能为空"
    mmap = {m.stem: m for m in mp_}
    missing = [d.name for d in dp_ if d.stem not in mmap]
    if missing:
        return (f"### ⚠️ {len(missing)} 张缺陷图找不到同名掩膜:"
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    imgs = [_load(x) for x in np_]
    dimg = [_load(x) for x in dp_]
    dmsk = [_mask(mmap[d.stem]) for d in dp_]
    return _fit_core(imgs, dimg, dmsk, "自选文件")


def _fit_core(imgs, dimg, dmsk, product):
    # 重复fit的显存泄漏修复:每次fit新建一整套模型(EAD+DINOv2+WRN+SAM+类型头),
    # 旧检测器被STATE替换后PyTorch缓存分配器不还显存——实测几轮fit后7.9/8.2GB打满,
    # WSL2下CUDA溢出到Windows共享内存,locate从~90ms劣化到~890ms(10倍)。
    # 必须先显式释放旧模型再建新的。
    if STATE["det"] is not None:
        STATE.update(det=None, loop=None)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    det = CompetitionLargeDetector()
    t0 = time.time()
    # 用 ActiveLearningLoop 承载:它维护样本集,操作员反馈时可增量重拟合
    loop = ActiveLearningLoop(det, imgs, dimg, defect_masks=dmsk)
    sec = time.time() - t0
    STATE.update(det=det, loop=loop, product=product, fit_sec=sec)
    th = getattr(det.type_head, "ready", False)
    return (f"### ✅ 迁移学习完成\n"
            f"- 产品:**{product}** · 正常 {len(imgs)} 张 + 缺陷 {len(dimg)} 张\n"
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


def _truth_of(path):
    """真实标签:按文件所在产品目录的 answer.csv 查(<产品>/test/<文件> 结构)。"""
    p = Path(path)
    csvp = p.parent.parent / "answer.csv"
    if p.parent.name != "test" or not csvp.exists():
        return "?"
    with open(csvp, encoding="utf-8") as f:
        return {r[0]: r[1] for r in list(csv.reader(f))[1:]}.get(p.name, "?")


def run_test(paths, uploads=None, progress=None):
    det = STATE["det"]
    if det is None:
        return [], None, "### ⚠️ 请先在 ① 完成迁移学习"
    todo = [Path(x) for x in (paths or []) if Path(x).is_file()
            and Path(x).suffix.lower() in IMG_EXTS]
    # 上传的图(Windows文件对话框选的):无对照答案,只出判定不计准确率
    for u in (uploads or []):
        up = Path(getattr(u, "name", u))
        if up.suffix.lower() in IMG_EXTS:
            todo.append(up)
    if not todo:
        return [], None, "### 请先勾选或上传图片"
    gal, rows = [], []
    tp = fn = fp = tn = 0
    lats = []
    for path in todo:
        f = path.name
        img = _load(path)
        t0 = time.time(); o = det.locate(img); ms = (time.time() - t0) * 1000
        lats.append(ms)
        truth = _truth_of(path)
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
    with gr.Blocks(title="AOI 实时在线 AI 质检",
                   css="footer {display: none !important;}") as app:
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
            with gr.Accordion("方式二:不用目录结构,直接系统文件框选三组图", open=False):
                with gr.Row():
                    up_n = gr.File(label="正常图(可框选多张)", file_count="multiple",
                                   file_types=["image"])
                    up_d = gr.File(label="缺陷图(多张)", file_count="multiple",
                                   file_types=["image"])
                    up_m = gr.File(label="缺陷掩膜(与缺陷图同名)", file_count="multiple",
                                   file_types=["image"])
                btn_fit2 = gr.Button("用选中的文件开始迁移学习")
            fit_out = gr.Markdown()
            def _set_root(path):
                global ROOT
                d = Path(path).expanduser()
                if not d.is_dir():
                    return gr.update(), f"### ⚠️ 目录不存在:{d}"
                ROOT = d
                new_ps = products()
                return (gr.update(choices=new_ps, value=(new_ps[0] if new_ps else None)),
                        f"已切换到 `{d}`,找到 {len(new_ps)} 个产品目录",
                        gr.update(root_dir=str(d)), gr.update(root_dir=str(d)))
            prod.change(preview_fit, prod, [g1, g2, info])
            app.load(preview_fit, prod, [g1, g2, info])
            btn_fit.click(do_fit, [prod, nn, nd], fit_out)
            btn_fit2.click(do_fit_files, [up_n, up_d, up_m], fit_out)
        with gr.Tab("② 在线检测(计时)"):
            fe = gr.FileExplorer(root_dir=str(ROOT), glob="**/*.png", file_count="multiple",
                                 label="像资源管理器一样点开目录、鼠标勾选图片(test/ 是混合测试流)",
                                 height=320)
            up = gr.File(label="或点这里走系统文件选择框(可框选/Ctrl多选,本机任意图片)",
                         file_count="multiple", file_types=["image"])
            with gr.Row():
                btn_all = gr.Button("全选当前产品的 test/"); btn_run = gr.Button("开始检测", variant="primary")
            res_md = gr.Markdown()
            gal = gr.Gallery(label="检测结果(红=预测缺陷区域/预测框)", columns=4, height=520)
            tbl = gr.Dataframe(headers=["文件", "真实", "判定", "缺陷类型", "异常分", "框数", "ms", "结果"],
                               label="逐图明细")
            with gr.Row():
                one = gr.Dropdown([], label="回溯某张图的检测逻辑")
                btn_ex = gr.Button("回溯")
            ex_md = gr.Markdown()

            def _refresh(p):
                return gr.update(choices=files_of(p, "test"))
            prod.change(_refresh, prod, one)
            app.load(_refresh, prod, one)
            btn_all.click(lambda p: gr.update(value=[str(ROOT / p / "test" / f)
                                                     for f in files_of(p, "test")]), prod, fe)
            btn_run.click(run_test, [fe, up], [gal, tbl, res_md])
            btn_ex.click(explain_one, one, ex_md)
        with gr.Tab("③ 操作员反馈(自学习)"):
            gr.Markdown(
                "看到判错的图?选中它 → 标出真实情况 → 提交。系统会用 VLM 说出这是什么缺陷、"
                "把样本并入训练集**重训分割头并重标阈值**(真改模型参数);救回它要付出过高"
                "误报时会**如实告知救不回**,而不是牺牲整条产线。单轮约 4 分钟。")
            fb_fe = gr.FileExplorer(root_dir=str(ROOT), glob="**/*.png", file_count="single",
                                    label="鼠标选中那张判错的图", height=240)
            with gr.Row():
                fb_up = gr.File(label="或走系统文件选择框", file_count="single", file_types=["image"])
                fb_truth = gr.Radio(["缺陷(漏检了)", "正常(误报了)"], value="缺陷(漏检了)",
                                    label="操作员判定的真实情况")
            fb_btn = gr.Button("提交反馈 → 自学习", variant="primary")
            fb_img = gr.Image(label="反馈样本")
            fb_md = gr.Markdown()

            def _do_fb(fpath, upfile, truth):
                loop = STATE["loop"]
                if loop is None:
                    return None, "### ⚠️ 请先在 ① 完成迁移学习"
                if isinstance(fpath, list):                   # FileExplorer single 也可能给列表
                    fpath = fpath[0] if fpath else None
                if upfile is not None:                        # 上传的自选图优先
                    img = _load(getattr(upfile, "name", upfile))
                elif fpath and Path(fpath).is_file():
                    img = _load(fpath)
                else:
                    return None, "### 请选中一张图或上传图片"
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

            fb_btn.click(_do_fb, [fb_fe, fb_up, fb_truth], [fb_img, fb_md])
            # 换根目录的接线放在这里:输出里的 fe/fb_fe 到②③页签才存在
            btn_root.click(_set_root, root_tb, [prod, info, fe, fb_fe])

        with gr.Tab("④ 交付汇总"):
            gr.Markdown(SUMMARY)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--root", default="demo_data")
    a = ap.parse_args()
    ROOT = Path(a.root)
    build().launch(server_name="0.0.0.0", server_port=a.port, share=False, show_api=False)
