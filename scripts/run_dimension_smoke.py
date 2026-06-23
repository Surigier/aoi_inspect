"""尺寸分支验证:python scripts/run_dimension_smoke.py
合成"暗背景+居中亮物体",物体尺寸变化(偏大/偏小)应被 DimensionADBranch 检出,
同尺寸(含轻微抖动)≈基线。坐实它覆盖"尺寸偏差"缺陷类型。"""
import random
import torch
from aoi.branches.dimension_ad import DimensionADBranch


def make(size, rng):
    """暗背景(0.1)+居中亮方块(0.8),边长=size(+抖动)。返回 (3,128,128)。"""
    img = torch.full((3, 128, 128), 0.1)
    s = size + rng.randint(-2, 2)
    o = (128 - s) // 2
    img[:, o:o + s, o:o + s] = 0.8
    return img


def main():
    rng = random.Random(0)
    fit = torch.stack([make(60, rng) for _ in range(20)])
    br = DimensionADBranch()
    br.fit(fit)

    def avg(size, n=8):
        return sum(br.infer(make(size, rng).unsqueeze(0)).score for _ in range(n)) / n

    print(f"{'物体尺寸':12s} {'尺寸分异常分':>12}")
    print(f"{'60(正常)':12s} {avg(60):12.2f}")
    for s in [45, 75, 90]:
        tag = f"{s}({'偏小' if s < 60 else '偏大'})"
        print(f"{tag:12s} {avg(s):12.2f}")
    print("\n期望:偏大/偏小尺寸异常分显著高于正常")


if __name__ == "__main__":
    main()
