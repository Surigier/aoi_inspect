"""图级AUROC对比:PatchCore/UniAD论文原文数字 vs 本系统在同一批MVTec类目上的真实测量。

论文数字来源(核实自论文PDF原文,不是网上转述):
- PatchCore(arXiv 2106.08265)Table S1,PatchCore-1%配置(附录,image-level AUROC):
  Cable 99.3 / Hazelnut 100 / Pill 97.0
- UniAD(arXiv 2206.03687)Table 1,"Ours"列(unified/separate两种设置):
  Cable 95.2/97.6 / Hazelnut 99.8/99.9 / Pill 93.7/88.3

**重要协议差异,不能假装完全可比**:论文用MVTec AD官方train/test划分,每类全量正常图
(通常200+张)训练;本系统用赛题协议(100正常+30缺陷现场少样本迁移)。数据集相同、
训练数据量级不同,只是同一数据集上的参考对照,不是同一协议下的严格对比。

用法:PYTHONPATH=. python scripts/run_auroc_compare.py
"""
import numpy as np

from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_mvtec


def roc_auc_score(labels, scores):
    """标准Mann-Whitney U形式,不依赖sklearn(2070上未装):
    AUROC = P(正样本分数 > 负样本分数),含并列分数的0.5权重处理,与
    sklearn.metrics.roc_auc_score数值等价。"""
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))

CATS = [
    ("hazelnut", ["crack", "cut", "hole"]),
    ("cable", ["missing_cable", "missing_wire"]),
    ("pill", ["color"]),
]


def main():
    for cat, folders in CATS:
        normals, fit_i, fit_m, test_defs, test_goods = prep_mvtec(cat, folders)
        det = CompetitionLargeDetector()
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        scores, labels = [], []
        for img, _ in test_defs:
            scores.append(det.frame_score(img)); labels.append(1)
        for img, _ in test_goods:
            scores.append(det.frame_score(img)); labels.append(0)
        auroc = roc_auc_score(labels, scores)
        print(f"RESULT {cat} auroc={auroc:.4f} n_defect={sum(labels)} "
              f"n_normal={len(labels) - sum(labels)}", flush=True)


if __name__ == "__main__":
    main()
