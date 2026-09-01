"""参数量对比图,数据来源见docs/delivery/结果报告.md和本文件注释,全部可复现:
- PatchCore骨干(WideResNet50-2完整参数量): timm.create_model('wide_resnet50_2')本地实测
- UniAD骨干(EfficientNet-B4): timm.create_model('efficientnet_b4')本地实测;
  transformer decoder部分论文未公开确切参数量,不编造,图中只标骨干+文字注明
- EfficientAD/本方案EAD核心: aoi/efficientad.py同架构本地实测(教师269万+学生427万×N+AE110万)
- 本方案WRN浅层/DINOv2/StructuralAD骨干/MobileSAM: 见结果报告"模型参数量"章节实测值

用法:PYTHONPATH=. python docs/delivery/figures/plot_params.py
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


HERO = "#2E5EAA"     # 本方案强调色
BASE = "#B0B7C3"     # 对照方法灰
ACCENT = "#D98C3D"

# ---------------- Fig 1: detection-backbone size comparison ----------------
fig, ax = plt.subplots(figsize=(4.2, 2.6))
labels = ["PatchCore\n(WideResNet50-2)", "UniAD\n(EfficientNet-B4 backbone)", "Ours\n(EAD core, 2-student ensemble)"]
values = [66.8, 17.5, 12.3]
colors = [BASE, BASE, HERO]
y = np.arange(len(labels))
bars = ax.barh(y, values, color=colors, height=0.55)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Parameters (millions)")
ax.set_title("Detection backbone size", fontsize=10, pad=8)
for b, v in zip(bars, values):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}M",
            va="center", fontsize=8)
ax.text(0.98, -0.28, "UniAD value is backbone-only; the transformer decoder's parameter count is not reported in the paper",
        transform=ax.transAxes, ha="right", fontsize=6, color="#666666")
fig.tight_layout()
save_pub(fig, "docs/delivery/figures/fig1_backbone_params")
plt.close(fig)

# ---------------- Fig 2: our system's module breakdown ----------------
fig, ax = plt.subplots(figsize=(4.2, 2.8))
mods = ["EAD detection core\n(2-student ensemble)", "WRN shallow layers\n(segmentation backbone)", "DINOv2 ViT-S/14\n(image-level gate)",
        "StructuralAD backbone\n(defect-type branch)", "MobileSAM\n(boundary refinement)"]
vals = [12.3, 4.1, 22.1, 24.9, 10.1]
order = np.argsort(vals)[::-1]
mods = [mods[i] for i in order]; vals = [vals[i] for i in order]
y = np.arange(len(mods))
bars = ax.barh(y, vals, color=HERO, height=0.55, alpha=0.85)
ax.set_yticks(y); ax.set_yticklabels(mods, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Parameters (millions)")
total = sum(vals)
ax.set_title(f"Module breakdown of our system (total ~{total:.0f}M)", fontsize=10, pad=8)
for b, v in zip(bars, vals):
    ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}M",
            va="center", fontsize=8)
fig.tight_layout()
save_pub(fig, "docs/delivery/figures/fig2_module_breakdown")
plt.close(fig)

print("已导出 fig1_backbone_params.{svg,pdf,png} 与 fig2_module_breakdown.{svg,pdf,png}")
