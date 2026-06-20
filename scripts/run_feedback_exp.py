"""用户反馈优化实验:python scripts/run_feedback_exp.py [visa类别=pcb3]
验证赛题第三支柱。稳健效果:操作员标记**误报的正常图**反馈→入记忆库→留出正常集误报率单调下降
(固定阈值,隔离阈值噪声,体现"模型层面"在线改进)。"""
import sys
import random
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from eval.mvtec import _load_img


def load_visa(cat, size=320):
    base = Path("data/visa") / cat / "Data" / "Images"
    n = [_load_img(p, size) for p in sorted((base / "Normal").glob("*.JPG"))]
    a = [_load_img(p, size) for p in sorted((base / "Anomaly").glob("*.JPG"))]
    return n, a


def main(cat="pcb3"):
    normal, anom = load_visa(cat)
    rng = random.Random(0)
    rng.shuffle(normal)
    rng.shuffle(anom)
    fit_n, fb_n, test_n = normal[:100], normal[100:250], normal[250:450]

    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    br = TextureADBranch(backbone=bb)
    ad = FewShotAdapter(br)
    ad.fit_fewshot(fit_n, anom[:30])
    thr = ad.threshold                       # 固定阈值,只看记忆库改进带来的误报变化

    def fp_rate(imgs):
        return sum(float(br.infer(x.unsqueeze(0)).score) >= thr for x in imgs) / len(imgs)

    print(f"visa/{cat}  起始误报率(留出正常集)={fp_rate(test_n):.3f}")
    fed = 0
    for _ in range(4):
        fps = [x for x in fb_n if float(br.infer(x.unsqueeze(0)).score) >= thr][:30]
        if not fps:
            break
        ids = {id(x) for x in fps}
        fit_n = fit_n + fps                  # 操作员标记的误报正常图 → 入库
        br.fit(torch.stack(fit_n))
        fb_n = [x for x in fb_n if id(x) not in ids]
        fed += len(fps)
        print(f"  反馈 {fed:3d} 张误报正常图后  误报率={fp_rate(test_n):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pcb3")
