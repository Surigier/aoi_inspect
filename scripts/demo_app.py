"""AOI 质检交互 Demo:python scripts/demo_app.py data/mvtec/bottle
检测(热力图+判决)→ 操作员标注 → 一键提交反馈即在线学习并重检(实时纠错演示)。

跑的是**生产检测器** CompetitionLargeDetector 本体,不是另搭一套简化管线。
唯一的让步是 train_steps 调小(见 DEMO_STEPS):生产默认 10000 步一次 fit 约 20 分钟,
每点一次反馈还要再等一次重拟合,交互演示等不起。**精度数字一律以成绩单脚本为准,
不要拿这个 Demo 的表现说事。**"""
import sys
import numpy as np
import torch
import gradio as gr
from aoi.competition import CompetitionLargeDetector
from aoi.active_learning import ActiveLearningLoop
from aoi.viz import overlay_heatmap
from eval.mvtec import load_category

LOOP = None
DEMO_STEPS = 400        # 仅为交互流畅;生产是 10000(见模块 docstring)


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB").resize((320, 320)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _verdict_md(o):
    head = "## 🔴 判定:缺陷" if o["is_defect"] else "## 🟢 判定:正常"
    thr = LOOP.adapter.threshold
    line = f"{head}\n\n异常分 **{o['score']:.3f}** / 阈值 {thr:.3f} · 类型:{o['defect_type']}"
    if o["is_defect"] and o.get("boxes"):
        line += f" · 检测框 {len(o['boxes'])} 个"
    return line


def predict_fn(pil):
    if pil is None:
        return None, "### 请先上传图片"
    t = _to_tensor(pil)
    o = LOOP.adapter.locate(t)                      # 生产推理入口(热力图/掩膜/框全在这)
    overlay = overlay_heatmap(t, o["anomaly_map"]) if o.get("anomaly_map") is not None else pil
    return overlay, _verdict_md(o)


def feedback_fn(pil, actual):
    if pil is None or actual is None:
        return None, "### 请先上传图片并选择实际情况", ""
    LOOP.feedback(_to_tensor(pil), is_defect=(actual == "缺陷"))   # 在线学习(动态调整模型参数)
    overlay, verdict = predict_fn(pil)                            # 立即重检,展示纠错
    status = (f"✅ 已学习「{actual}」并更新模型 · 样本库 正常 {len(LOOP.normals)} / 缺陷 {len(LOOP.defects)}"
              "(上方为反馈后的最新判定)")
    return overlay, verdict, status


def main(root):
    global LOOP
    data = load_category(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = CompetitionLargeDetector(device=device, train_steps=DEMO_STEPS)
    print("正在做少样本现场迁移(100 正常 + 30 缺陷),首次启动需要几分钟…", flush=True)
    LOOP = ActiveLearningLoop(det, data["train_normal"][:100], data["test_defect"][:30])

    with gr.Blocks(title="AOI 实时质检") as demo:
        gr.Markdown(
            "# 🔍 AOI 实时在线 AI 质检 Demo\n"
            "EfficientAD + DINOv2 双判据检测 · 监督分割头 + SAM 精化定位 · "
            "VLM 蒸馏的缺陷类型归属 · **操作员反馈 → 在线更新(误检/漏检都走实时快路径)**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="① 上传待检图")
                detect_btn = gr.Button("② 检测", variant="primary")
                gr.Markdown("—— 若判错,操作员介入 ——")
                actual = gr.Radio(["正常", "缺陷"], label="③ 标注实际情况")
                fb_btn = gr.Button("④ 提交反馈 → 在线学习并重检", variant="secondary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="异常热力图(红=可疑)")
                verdict = gr.Markdown("### 等待检测…")
                status = gr.Markdown("")
        detect_btn.click(predict_fn, inp, [out_img, verdict])
        fb_btn.click(feedback_fn, [inp, actual], [out_img, verdict, status])

    demo.launch()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/mvtec/bottle")
