"""模拟考题:python scripts/mock_exam.py <category_dir> [更多目录...]
把给定类别当"未知产品",按官方协议(100正常+30缺陷现场迁移→测剩余)跑,
套用竞赛评分权重(完整度50%+准确率20%+时间30%)估算竞赛得分。
完整度为主观自评;准确率/时间为客观实测。多目录则取均值。"""
import sys
import random
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category

COMPLETENESS = 95.0   # 主观自评:多分支+视频+少样本+反馈闭环+文档,系统完整


def exam_one(bb, root):
    data = load_category(root)
    rng = random.Random(0)
    nm, df = data["train_normal"][:], data["test_defect"][:]
    rng.shuffle(nm)
    rng.shuffle(df)
    fn, fd = nm[:min(100, len(nm))], df[:min(30, len(df) // 2)]
    ti = data["test_normal"] + df[len(fd):]
    tl = [0] * len(data["test_normal"]) + [1] * len(df[len(fd):])
    adapter = MultiBranchAdapter([TextureADBranch(backbone=bb), StructuralADBranch(backbone=bb, grid_size=16)])
    m = run_protocol(adapter, fn, fd, ti, tl)
    return m, len(ti)


def main(roots):
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    aurocs, accs, lats = [], [], []
    for root in roots:
        m, n = exam_one(bb, root)
        aurocs.append(m["auroc"])
        accs.append(m["accuracy"])
        lats.append(m["latency_ms_mean"])
        print(f"{root:32s} AUROC={m['auroc']:.3f} 准确率={m['accuracy']:.3f} 延时={m['latency_ms_mean']:.0f}ms (test {n})")

    auroc = sum(aurocs) / len(aurocs)
    acc = sum(accs) / len(accs)
    lat = sum(lats) / len(lats)
    # 竞赛子项(0-100):准确率分用 AUROC(阈值无关,更稳);时间分 <200ms 满分
    acc_score = auroc * 100
    time_score = 100.0 if lat <= 200 else max(0.0, 100 * (2.0 - lat / 200))
    competition = 0.5 * COMPLETENESS + 0.2 * acc_score + 0.3 * time_score
    print("\n=== 模拟竞赛得分(估算)===")
    print(f"均值: AUROC={auroc:.3f} 准确率={acc:.3f} 延时={lat:.0f}ms")
    print(f"完整度(自评)={COMPLETENESS:.0f} | 准确率分(=AUROC)={acc_score:.0f} | 时间分={time_score:.0f}")
    print(f"竞赛得分 = 0.5×{COMPLETENESS:.0f} + 0.2×{acc_score:.0f} + 0.3×{time_score:.0f} = {competition:.1f}/100")
    print(f"竞赛占总分 60% → 折合 {competition * 0.6:.1f}/60;专家分(40%)另计。")


if __name__ == "__main__":
    main(sys.argv[1:])
