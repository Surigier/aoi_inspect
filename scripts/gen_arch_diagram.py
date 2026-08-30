#!/usr/bin/env python3
"""生成系统四层级联架构图(检测/定位/归因/反馈),供技术方案说明书§3.1插图。
所有模块名称与参数取自 aoi/competition.py 实际实现与结果报告.md,不新造任何描述。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# 字体:用单一的"Noto Sans SC"(中英文+符号全覆盖),不做多字体拼接回退。
# 踩过的坑:matplotlib的按字符回退列表(如 ["DejaVu Sans", 中文字体]) 对**旋转文本**
# 不生效——本机唯一现成的中文字体 Droid Sans Fallback 本身只有CJK表意字符、
# 一个拉丁字母/数字/符号都没有(用fontTools核实过),要跟DejaVu Sans拼接才能凑齐
# 一整句"中文+英文+数字+箭头"的文字,而这个拼接机制在旋转90°的文字上失效,
# 渲染成方块占位符。改用单一字体从根上消掉"多字体拼接"这个不稳定环节。
import matplotlib.font_manager as fm
_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fonts", "NotoSansSC.otf")
_FONT_URL = ("https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
             "SimplifiedChinese/NotoSansCJKsc-Regular.otf")
if not os.path.exists(_FONT_PATH):
    os.makedirs(os.path.dirname(_FONT_PATH), exist_ok=True)
    import urllib.request
    print(f"本地未找到字体,尝试下载: {_FONT_URL}")
    urllib.request.urlretrieve(_FONT_URL, _FONT_PATH)
fm.fontManager.addfont(_FONT_PATH)
CJK = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["font.sans-serif"] = [CJK]
plt.rcParams["axes.unicode_minus"] = False
print("使用字体:", CJK, f"({_FONT_PATH})")

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(0, 100)
ax.set_ylim(0, 148)
ax.axis("off")

COL = dict(
    input="#e8ecf3", detect="#dfeeff", locate="#e3f6ea",
    attr="#fdf0dc", fb="#fbe4e4", gate="#fff6cf", border="#333333",
)


def box(x, y, w, h, text, fc, fs=10.5, weight="normal", ec="#333333", lw=1.3, ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, zorder=3, linespacing=1.5)
    return b


def arrow(x1, y1, x2, y2, text=None, style="-|>", color="#333333", lw=1.6,
          connectionstyle="arc3,rad=0.0", fs=9.5, text_dx=3.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                         lw=lw, connectionstyle=connectionstyle, zorder=1,
                         mutation_scale=14)
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + text_dx, (y1 + y2) / 2, text, ha="left", va="center",
                fontsize=fs, color=color)


# ---------- 标题 ----------
ax.text(50, 145, "少样本工业异常检测级联系统总体架构", ha="center", fontsize=16, fontweight="bold")
ax.text(50, 141.5, "locate(img)：检测层判正常立即早退 → 定位层 → 归因层；反馈层独立于该推理主路径",
        ha="center", fontsize=10, color="#555555")

# ---------- 输入 ----------
box(35, 132, 30, 6, "输入图像\n(2500×2500，单次GPU上传)", COL["input"], fs=10.5)
arrow(50, 132, 50, 127.5)

# ---------- 检测层 ----------
box(5, 108, 90, 18, "", COL["gate"], ec="#b8960b", lw=1.5)
ax.text(50, 124, "① 检测层  predict(img)  ——  唯一决定“是不是缺陷”", ha="center",
        fontsize=11.5, fontweight="bold", color="#7a5c00")

box(10, 111, 34, 10,
    "EfficientAD 师生蒸馏\n(教师冻结 + 双学生集成)\n→ 检测分 zEAD", COL["detect"], fs=9.7)
box(56, 111, 34, 10,
    "DINOv2 图级协同检测门\n(自监督特征，独立判据)\n→ 检测分 zDINO", COL["detect"], fs=9.7)
ax.text(50, 116, "受控平等\n融合", ha="center", fontsize=9.3, color="#7a5c00")
arrow(44, 116, 47.5, 116)
arrow(56, 116, 52.5, 116)
ax.text(50, 109.3, "判决 = max(zEAD, zDINO) ≥ 阈值  （创新点③：正交双判据，取最大值而非加权平均）",
        ha="center", fontsize=8.8, color="#7a5c00")

arrow(50, 108, 50, 103.5, text="判为异常", fs=9.5)
box(58, 100, 34, 5.5, "判为正常 → 立即返回（早退，不触发定位/归因）", "#f5f5f5",
    fs=8.8, ec="#999999", ls="--")
arrow(38, 105.5, 58, 102.8, color="#999999", lw=1.2, connectionstyle="arc3,rad=-0.15")

# ---------- 定位层 ----------
box(5, 78, 90, 22, "", COL["locate"], ec="#2e7d46", lw=1.5)
ax.text(50, 97.3, "② 定位层  segment(img)  ——  缺陷在哪、多大范围", ha="center",
        fontsize=11.5, fontweight="bold", color="#1f5c31")

box(10, 88.5, 26, 7, "WRN50 浅层(1,2)\n特征提取 @512", COL["locate"], fs=9.3)
arrow(36, 92, 39.5, 92)
box(39.5, 88.5, 26, 7, "双头联合训练\n监督分割头\n(linear+conv联合)", COL["locate"], fs=9.0)
arrow(65.5, 92, 69, 92)
box(69, 88.5, 26, 7, "MobileSAM\n受控边界精化\n(逐区域OOF门控)", COL["locate"], fs=9.0)

ax.text(50, 84.5, "创新点：浅层特征(非深层语义)，含漏检IoU 0.305→0.449(+47%)，8ms vs 36ms",
        ha="center", fontsize=8.8, color="#1f5c31")
ax.text(50, 80.8, "唯一通过零回退验证的候选：双头联合训练 → acc+0.010 / IoU+0.005 / 框命中+0.028",
        ha="center", fontsize=8.8, color="#1f5c31")

arrow(50, 78, 50, 73.5)

# ---------- 归因层 ----------
box(5, 50, 90, 23, "", COL["attr"], ec="#b85c00", lw=1.5)
ax.text(50, 70.3, "③ 归因层  ——  属于赛题5类中的哪一类（创新点①②）", ha="center",
        fontsize=11.5, fontweight="bold", color="#8a4400")

box(9, 60, 40, 8.5,
    "fit期(不计时)：\nVLM双图对比标注30张缺陷图\n(待判图 + 同坐标正常参考图)\n→ 蒸馏为质心分类表",
    "#ffe8c2", fs=9.0, ec="#b85c00")
arrow(49, 64, 53, 64, text="蒸馏", fs=9)
box(53, 60, 38, 8.5,
    "推理期：位置匹配特征\n+ 最近质心分类\n(零API / 零外网依赖)", COL["attr"], fs=9.3)

ax.text(50, 56.5, "端到端类型归属：启发式基线38% → 双图归因范式74%（GT掩膜上界87%）",
        ha="center", fontsize=8.8, color="#8a4400")
ax.text(50, 52.8, "任一环（无key/无网/超时/解析失败）断开 → 自动降级启发式，检测定位不受影响",
        ha="center", fontsize=8.5, color="#8a4400")

# ---------- 输出 ----------
arrow(50, 50, 50, 45.5)
box(28, 39, 44, 6, "输出：是否缺陷 / 定位框 / 缺陷类型 / 延时", COL["input"], fs=10)

# ---------- 反馈层(独立分支) ----------
box(5, 12, 90, 22, "", COL["fb"], ec="#a02020", lw=1.5, ls="--")
ax.text(50, 31.3, "④ 反馈层  ActiveLearningLoop  ——  独立于推理主路径（创新点④）", ha="center",
        fontsize=11.5, fontweight="bold", color="#7a1818")
box(9, 21, 26, 7.5, "操作员标注\n漏检 / 误检", "#fbe4e4", fs=9.3, ec="#a02020")
arrow(35, 24.7, 39, 24.7)
box(39, 21, 26, 7.5, "允许的两类改变：\n①分割头增量重训\n②阈值确定性硬约束", COL["fb"], fs=8.8)
arrow(65, 24.7, 69, 24.7)
box(69, 21, 24, 7.5, "VLM即时诊断\n(一句话现象+类型)", "#fbe4e4", fs=9.0, ec="#a02020")
ax.text(50, 16.5, "其余标定(阈值重投票/延时自适应裁剪)在反馈期全部冻结，仅初始迁移时执行一次",
        ha="center", fontsize=8.6, color="#7a1818")
ax.text(50, 13.3, "六轮端到端留出验证收敛；反馈单轮耗时 1193s → 251s（跳过EAD学生重训）",
        ha="center", fontsize=8.6, color="#7a1818")

# 反馈层与主路径的虚线连接(箭头指向定位层,表示反馈向上更新分割头权重/阈值,
# 而非反馈层本身在推理主路径上)
arrow(5, 23, 5, 89, color="#a02020", lw=1.1, style="-|>",
      connectionstyle="arc3,rad=0.25")
ax.text(1.5, 60, "增量更新模型参数", ha="center", va="center", fontsize=8,
        color="#a02020", rotation=90)

plt.tight_layout()
out = "docs/delivery/架构图.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("已保存:", out)
