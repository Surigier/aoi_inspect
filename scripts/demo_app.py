"""AOI 质检 Demo:python scripts/demo_app.py data/mvtec/bottle
上传图 → 检测(热力图+判决)→ 操作员标注 → 提交反馈在线更新。"""
import sys
import numpy as np
import torch
import gradio as gr
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.multibranch import MultiBranchAdapter
from aoi.active_learning import ActiveLearningLoop
from aoi.viz import overlay_heatmap
from eval.mvtec import load_category

LOOP = None


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def predict_fn(pil):
    t = _to_tensor(pil)
    r, is_def = LOOP.predict(t.unsqueeze(0))
    overlay = overlay_heatmap(t, r.anomaly_map) if r.anomaly_map is not None else pil
    return overlay, f"{'缺陷' if is_def else '正常'}  (score={r.score:.3f}, type={r.defect_type})"


def feedback_fn(pil, actual):
    LOOP.feedback(_to_tensor(pil), is_defect=(actual == "缺陷"))
    return f"已反馈「{actual}」;正常库={len(LOOP.normals)} 缺陷库={len(LOOP.defects)}。请重新点击「检测」查看更新效果。"


def main(root):
    global LOOP
    data = load_category(root)
    bb = Backbone(pretrained=True, device="cuda")
    adapter = MultiBranchAdapter([
        TextureADBranch(backbone=bb, coreset_ratio=0.25),
        StructuralADBranch(backbone=bb, grid_size=16),
    ])
    LOOP = ActiveLearningLoop(adapter, data["train_normal"][:100], data["test_defect"][:30])
    with gr.Blocks() as demo:
        gr.Markdown("# AOI 实时质检 Demo:检测 + 用户反馈闭环")
        with gr.Row():
            inp = gr.Image(type="pil", label="上传待检图")
            out_img = gr.Image(label="异常热力图")
        out_lbl = gr.Textbox(label="判决")
        gr.Button("检测").click(predict_fn, inp, [out_img, out_lbl])
        actual = gr.Radio(["正常", "缺陷"], label="操作员标注实际情况")
        fb_out = gr.Textbox(label="反馈结果")
        gr.Button("提交反馈(在线更新)").click(feedback_fn, [inp, actual], fb_out)
    demo.launch()


if __name__ == "__main__":
    main(sys.argv[1])
