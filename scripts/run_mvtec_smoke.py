"""真实 MVTec 冒烟:python scripts/run_mvtec_smoke.py data/mvtec/bottle
按官方协议抽 100 正常 + 30 缺陷迁移,在剩余测试样本上输出 AUROC/准确率/延时。"""
import sys
import random
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category


def main(root):
    data = load_category(root)
    rng = random.Random(0)
    normals = data["train_normal"][:]
    defects = data["test_defect"][:]
    rng.shuffle(normals)               # 带种子洗牌再抽样,避免按文件名顺序的偏样本
    rng.shuffle(defects)
    fit_normal = normals[:100]
    fit_defect = defects[:30]
    test_imgs = data["test_normal"] + defects[30:]
    test_labels = [0] * len(data["test_normal"]) + [1] * len(defects[30:])
    branch = TextureADBranch(backbone=Backbone(pretrained=True, device="cuda"), coreset_ratio=0.25)
    adapter = FewShotAdapter(branch)
    print(run_protocol(adapter, fit_normal, fit_defect, test_imgs, test_labels))


if __name__ == "__main__":
    main(sys.argv[1])
