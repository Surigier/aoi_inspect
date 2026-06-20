"""标准三分支 ensemble 工厂(供 submit / demo / 脚本统一复用)。
纹理(记忆库,外观)+ 结构(位置感知,缺件/逻辑)+ 判别头(监督,用现场缺陷抬上限)。"""
from .backbone import Backbone
from .branches.texture_ad import TextureADBranch
from .branches.structural_ad import StructuralADBranch
from .branches.disc_ad import DiscriminativeBranch
from .multibranch import MultiBranchAdapter


def default_adapter(backbone: Backbone, grid_size: int = 16) -> MultiBranchAdapter:
    return MultiBranchAdapter([
        TextureADBranch(backbone=backbone),
        StructuralADBranch(backbone=backbone, grid_size=grid_size),
        DiscriminativeBranch(backbone=backbone),
    ])
