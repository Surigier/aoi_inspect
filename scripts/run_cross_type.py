"""跨缺陷类型泛化测试:python scripts/run_cross_type.py data/mpdd/metal_plate
只用**一种**缺陷类型训练,测**其他没见过的**缺陷类型。
对比 纹理(无监督)/ 判别头(监督)/ 融合 —— 看判别头是否过拟合到见过的类型。"""
import sys
import random
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.disc_ad import DiscriminativeBranch
from aoi.fewshot import FewShotAdapter
from aoi.ensemble import default_adapter
from eval.protocol import run_protocol, image_auroc
from eval.mvtec import _load_img
import numpy as np


def main(root):
    root = Path(root)
    size = 320
    train_normal = [_load_img(p, size) for p in sorted((root / "train" / "good").glob("*.png"))]
    test_normal = [_load_img(p, size) for p in sorted((root / "test" / "good").glob("*.png"))]
    types = [d for d in sorted((root / "test").iterdir()) if d.is_dir() and d.name != "good"]
    by_type = {d.name: [_load_img(p, size) for p in sorted(d.glob("*.png"))] for d in types}
    names = list(by_type)
    seen = names[0]                                  # 只训练第一种缺陷
    unseen = names[1:]                               # 测其余没见过的
    fit_def = by_type[seen][:min(30, len(by_type[seen]))]
    test_def = [x for n in unseen for x in by_type[n]]
    print(f"训练缺陷类型: {seen}({len(fit_def)})  |  测试未见类型: {unseen}({len(test_def)})")

    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    fn = train_normal[:100]
    ti = test_normal + test_def
    tl = [0] * len(test_normal) + [1] * len(test_def)
    for name, ad in [("纹理(无监督)", FewShotAdapter(TextureADBranch(backbone=bb))),
                     ("判别头(监督)", FewShotAdapter(DiscriminativeBranch(backbone=bb, epochs=300, lr=0.05))),
                     ("融合", default_adapter(bb))]:
        try:
            if isinstance(ad, FewShotAdapter) and hasattr(ad.branch, "fit_supervised"):
                # 判别头单分支:用 FewShotAdapter 跑不通(它调 fit),改直接评
                br = ad.branch
                br.fit_supervised(fn, fit_def)
                sc = [br.infer(x.unsqueeze(0)).score for x in ti]
                au = image_auroc(np.array(sc), np.array(tl))
            else:
                au = run_protocol(ad, fn, fit_def, ti, tl)["auroc"]
            print(f"  {name:14s} 未见类型 AUROC={au:.3f}")
        except Exception as e:
            print(f"  {name:14s} 失败 {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/mpdd/metal_plate")
