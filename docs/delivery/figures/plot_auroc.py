"""AUROC对比图:本系统 vs PatchCore/UniAD论文原文数字,同一批MVTec类目(hazelnut/cable/pill)。

数据来源:
- 本系统:scripts/run_auroc_compare.py 真实实测(hazelnut 0.9701 / cable 1.0000 / pill 1.0000)
- PatchCore(arXiv 2106.08265)Table S1,PatchCore-1%配置
- UniAD(arXiv 2206.03687)Table 1,"Ours"列,separate(每类单独)设置

**重要协议差异,图上必须标注**:论文用MVTec AD官方划分、每类全量正常图(通常200+张)训练;
本系统用赛题协议(100正常+30缺陷现场少样本迁移)。数据集相同,训练数据量级不同。

用法:PYTHONPATH=. python docs/delivery/figures/plot_auroc.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def save_pub(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


HERO = "#2E5EAA"
BASE1 = "#B0B7C3"
BASE2 = "#8A93A3"

cats = ["Hazelnut", "Cable", "Pill"]
ours = [97.01, 100.00, 100.00]
patchcore = [100.0, 99.3, 97.0]
uniad = [99.9, 97.6, 88.3]   # separate (per-class) setting, closer to our "per-product fit" scenario

x = np.arange(len(cats))
w = 0.26

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.bar(x - w, patchcore, width=w, color=BASE1, label="PatchCore (full-shot)")
ax.bar(x, uniad, width=w, color=BASE2, label="UniAD (full-shot, separate)")
ax.bar(x + w, ours, width=w, color=HERO, label="Ours (100+30 few-shot)")

for xi, v in zip(x - w, patchcore):
    ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7)
for xi, v in zip(x, uniad):
    ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7)
for xi, v in zip(x + w, ours):
    ax.text(xi, v + 0.6, f"{v:.2f}", ha="center", fontsize=7, color=HERO, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Image-level AUROC (%)")
ax.set_ylim(80, 105)
ax.set_title("Image-level AUROC: reference comparison on shared MVTec categories", fontsize=9.5, pad=8)
ax.legend(loc="lower left", fontsize=7)
ax.text(0.5, -0.26,
        "Protocols are not equivalent: papers train on the full official normal-image set (200+ images);\n"
        "ours uses the competition protocol (100 normal + 30 defect images, on-site few-shot transfer).\n"
        "Same dataset, different training-data scale — shown for reference, not a like-for-like comparison.",
        transform=ax.transAxes, ha="center", fontsize=6.5, color="#666666")
fig.tight_layout()
save_pub(fig, "docs/delivery/figures/fig3_auroc_compare")
plt.close(fig)

print("已导出 fig3_auroc_compare.{svg,pdf,png}")
