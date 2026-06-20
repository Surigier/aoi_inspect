"""AOI 质检交互 Demo:python scripts/demo_app.py data/mvtec/bottle
检测(热力图+判决)→ 操作员标注 → 一键提交反馈即在线学习并重检(实时纠错演示)。
ensemble = 纹理(记忆库)+ 结构(位置感知)+ 判别头(监督)。"""
import sys
import numpy as np
import torch
import gradio as gr
from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.active_learning import ActiveLearningLoop
from aoi.viz import overlay_heatmap
from eval.mvtec import load_category

LOOP = None


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB").resize((320, 320)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _verdict_md(r, is_def):
    head = "## 🔴 判定:缺陷" if is_def else "## 🟢 判定:正常"
    return f"{head}\n\n异常分 **{r.score:.3f}** / 阈值 {LOOP.adapter.threshold:.3f} · 类型:{r.defect_type}"


def predict_fn(pil):
    if pil is None:
        return None, "### 请先上传图片"
    t = _to_tensor(pil)
    r, is_def = LOOP.predict(t.unsqueeze(0))
    overlay = overlay_heatmap(t, r.anomaly_map) if r.anomaly_map is not None else pil
    return overlay, _verdict_md(r, is_def)


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
    bb = Backbone(pretrained=True, device=device)
    LOOP = ActiveLearningLoop(default_adapter(bb), data["train_normal"][:100], data["test_defect"][:30])

    with gr.Blocks(title="AOI 实时质检") as demo:
        gr.Markdown(
            "# 🔍 AOI 实时在线 AI 质检 Demo\n"
            "纹理(记忆库)+ 结构(位置感知缺件)+ 判别头(监督)三分支融合 · "
            "少样本现场迁移 · **操作员反馈→在线调整模型参数**"
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
