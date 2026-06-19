from aoi.fusion import znorm, fuse, auroc

def test_auroc_perfect_and_reversed():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1]) == 0.0

def test_auroc_ties_give_half():
    # 全等分(完全无区分)应为 0.5,不受标签顺序影响(平均秩)
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5

def test_znorm_basic():
    assert znorm(2.0, mean=0.0, std=2.0) == 1.0

def test_znorm_zero_std_safe():
    # std 为 0 时退化为减均值(不除零)
    assert znorm(3.0, mean=1.0, std=0.0) == 2.0

def test_fuse_takes_max():
    assert fuse([0.2, 1.5, -0.3]) == 1.5
