"""标准 ensemble 工厂(供 submit / demo / 脚本统一复用)。覆盖赛题 5 类缺陷:
纹理(外观)+ 结构(位置感知,缺件/逻辑)+ 判别头(监督)+ 色彩(色度变化)。
可靠性加权软融合自动按类分配权重 → 无关分支被压、相关分支贡献。"""
from .backbone import Backbone
from .branches.texture_ad import TextureADBranch
from .branches.structural_ad import StructuralADBranch
from .branches.disc_ad import DiscriminativeBranch
from .branches.color_ad import ColorADBranch
from .multibranch import MultiBranchAdapter


def default_adapter(backbone: Backbone, grid_size: int = 16) -> MultiBranchAdapter:
    return MultiBranchAdapter([
        TextureADBranch(backbone=backbone),
        StructuralADBranch(backbone=backbone, grid_size=grid_size),
        DiscriminativeBranch(backbone=backbone),
        ColorADBranch(grid_size=grid_size),
    ])
