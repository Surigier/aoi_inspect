"""zero-shot 无样本验证:python scripts/run_zeroshot_smoke.py [类别...]
完全不给任何样本(无 fit),纯 CLIP 正常/异常文本提示判异常,报 AUROC。
坐实赛题"无样本(zero-shot)启动"能力。"""
import sys
import torch
from aoi.clip_encoder import CLIPEncoder
from aoi.branches.zeroshot_clip import ZeroShotCLIPBranch
from aoi.fusion import auroc
from eval.mvtec import load_category

CATS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut"]


def main():
    cats = sys.argv[1:] or CATS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = CLIPEncoder(device=dev)
    aus = []
    for cat in cats:
        try:
            data = load_category(f"data/mvtec/{cat}", size=224)
        except Exception as e:
            print(f"{cat}: skip ({e})"); continue
        br = ZeroShotCLIPBranch(enc, class_name=cat.replace("_", " "))   # 无 fit
        test = data["test_normal"] + data["test_defect"]
        lab = [0] * len(data["test_normal"]) + [1] * len(data["test_defect"])
        scores = [br.infer(im.unsqueeze(0)).score for im in test]
        au = auroc(scores, lab)
        aus.append(au)
        print(f"{cat:12s} zero-shot AUROC={au:.3f}  (n={len(test)}, 无样本)", flush=True)
    if aus:
        print(f"\n平均 zero-shot AUROC={sum(aus)/len(aus):.3f}(完全无训练样本)")


if __name__ == "__main__":
    main()
