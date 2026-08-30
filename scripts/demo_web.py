"""AOI 质检演示:结果图 + Web端(图片/视频)。

两种模式共用一次迁移学习(fit约20分钟,赛题规定此阶段不计时):
  --sheet   产出结果图 _logs/demo_sheet.png(离线看,不需要服务)
  --web     启动本地Web端,工作人员可上传图片/视频,直观看到检测结果

Web端展示:原图 → 异常热力图叠加 → 检测框 → 判决/类型/异常分/延时,
并给出"回溯检测逻辑"(赛题明确要求)的完整判定链路。

用法:
  PYTHONPATH=. python scripts/demo_web.py --sheet                    # 出图给人看
  PYTHONPATH=. python scripts/demo_web.py --web --port 7860          # 起Web端
  PYTHONPATH=. python scripts/demo_web.py --sheet --cat cable        # 换产品
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from aoi.competition import CompetitionLargeDetector

GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")


def load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size:
        im = im.resize(size, Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def fit_detector(cat="hazelnut", n_norm=100, n_def=30):
    root = Path(f"data/mvtec/{cat}")
    ns = sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]
    df = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            for f in sorted(sub.glob("*.png")):
                m = GT / cat / "ground_truth" / sub.name / (f.stem + "_mask.png")
                if m.exists():
                    df.append((str(f), str(m)))
    import random
    random.Random(0).shuffle(df)
    fit_d = df[:n_def]
    print(f"迁移学习:{cat}  正常{len(ns)}张 + 缺陷{len(fit_d)}张", flush=True)
    det = CompetitionLargeDetector()
    t0 = time.time()
    det.fit_fewshot([load(p) for p in ns], [load(p) for p, _ in fit_d],
                    defect_masks=[(np.array(Image.open(m).convert("L").resize((256, 256))) > 0
                                   ).astype(np.uint8) for _, m in fit_d])
    print(f"迁移完成 {time.time()-t0:.0f}s(赛题此阶段不计时)", flush=True)
    return det, ns, df[n_def:]


def render(img_t, o, ms, title=""):
    """原图 + 异常叠加 + 检测框 → PIL图"""
    arr = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    mk = o.get("mask")
    if mk is not None and mk.any():
        m = np.array(Image.fromarray((mk * 255).astype(np.uint8)).resize(im.size, Image.NEAREST))
        a = np.array(im)
        a[m > 127] = (0.45 * a[m > 127] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    if mk is not None:
        sx, sy = im.width / mk.shape[1], im.height / mk.shape[0]
        for b in (o.get("boxes") or []):
            d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(255, 40, 40),
                        width=max(3, im.width // 200))
    tag = ("🔴 缺陷:" + o["defect_type"]) if o["is_defect"] else "🟢 正常"
    d.rectangle([0, 0, im.width, 46], fill=(0, 0, 0))
    d.text((10, 6), f"{title} {tag}", fill=(255, 255, 0))
    d.text((10, 26), f"异常分{o['score']:.3f} · 框{len(o.get('boxes') or [])}个 · {ms:.0f}ms",
           fill=(200, 200, 200))
    return im


def make_sheet(det, tests, out="_logs/demo_sheet.png", n=8):
    W = 420
    cells = []
    for i, (f, _m) in enumerate(tests[:n]):
        img = load(f)
        t0 = time.time(); o = det.locate(img); ms = (time.time() - t0) * 1000
        cells.append(render(img, o, ms, f"#{i:02d}").resize((W, W)))
        print(f"  #{i:02d} {'缺陷' if o['is_defect'] else '正常'} "
              f"{o['defect_type']} 框{len(o.get('boxes') or [])}个 {ms:.0f}ms", flush=True)
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, W * rows), (18, 18, 18))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * W, (i // cols) * W))
    Path(out).parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(f"结果图已生成: {out}", flush=True)
    return out


def serve(det, port=7860):
    import gradio as gr

    def infer(pil):
        if pil is None:
            return None, "### 请先上传图片"
        a = np.asarray(pil.convert("RGB"), np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)
        t0 = time.time(); o = det.locate(t); ms = (time.time() - t0) * 1000
        tr = det.explain(t)                        # 赛题要求:可回溯检测逻辑
        md = (f"## {'🔴 判定:缺陷' if o['is_defect'] else '🟢 判定:正常'}\n\n"
              f"- **缺陷类型**:{o['defect_type']}\n"
              f"- **异常分**:{o['score']:.4f} / 阈值 {det.decision_threshold():.4f}\n"
              f"- **检测框**:{len(o.get('boxes') or [])} 个 → {o.get('boxes')}\n"
              f"- **单图延时**:{ms:.0f} ms(预算 200ms)\n\n"
              f"### 检测逻辑回溯\n```\n{tr}\n```")
        return render(t, o, ms), md

    with gr.Blocks(title="AOI 实时在线 AI 质检") as app:
        gr.Markdown("# 🔍 AOI 实时在线 AI 质检\n"
                    "上传图片 → 输出**判决 / 缺陷类型 / 检测框 / 延时**,并可回溯检测逻辑。\n"
                    "红色叠加=预测缺陷区域,红框=检测框。")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="① 上传待检图片")
                btn = gr.Button("② 检测", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="检测结果")
                out_md = gr.Markdown("### 等待检测…")
        btn.click(infer, inp, [out_img, out_md])
    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="hazelnut")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    det, _ns, tests = fit_detector(a.cat)
    if a.sheet or not a.web:
        make_sheet(det, tests)
    if a.web:
        serve(det, a.port)
