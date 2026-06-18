from aoi.fusion import znorm, fuse

def test_znorm_basic():
    assert znorm(2.0, mean=0.0, std=2.0) == 1.0

def test_znorm_zero_std_safe():
    # std 为 0 时退化为减均值(不除零)
    assert znorm(3.0, mean=1.0, std=0.0) == 2.0

def test_fuse_takes_max():
    assert fuse([0.2, 1.5, -0.3]) == 1.5
