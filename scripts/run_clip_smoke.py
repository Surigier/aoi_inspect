"""零样本 CLIP 冒烟:python scripts/run_clip_smoke.py data/mvtec/bottle [class_name]
用 CLIP 文本提示零样本检测,按官方协议输出 AUROC/准确率/延时。"""
import os
import sys
from aoi.clip_encoder import CLIPEncoder
from aoi.branches.zeroshot_clip import ZeroShotCLIPBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category


def main(root, class_name="object"):
    data = load_category(root)
    encoder = CLIPEncoder(device="cuda")
    branch = ZeroShotCLIPBranch(encoder, class_name=class_name)
    adapter = FewShotAdapter(branch)
    # 零样本:fit 为空操作;少量样本仅用于阈值标定
    fit_normal = data["test_normal"][:10]
    fit_defect = data["test_defect"][:10]
    test_imgs = data["test_normal"][10:] + data["test_defect"][10:]
    test_labels = [0] * len(data["test_normal"][10:]) + [1] * len(data["test_defect"][10:])
    print(run_protocol(adapter, fit_normal, fit_defect, test_imgs, test_labels))


if __name__ == "__main__":
    root = sys.argv[1]
    # 默认用目录名当类名(如 data/mvtec/metal_nut → "metal_nut"),提示词更贴切
    class_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(root.rstrip("/"))
    main(root, class_name)
