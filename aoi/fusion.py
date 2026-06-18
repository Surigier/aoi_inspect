def znorm(score: float, mean: float, std: float) -> float:
    """按正常分的均值/方差做 z 归一化;std≈0 时退化为减均值,避免除零。"""
    return (score - mean) / (std if std > 1e-12 else 1.0)


def fuse(norm_scores) -> float:
    """多分支归一化分数融合:取最大(任一分支报异常即视为异常)。"""
    return max(norm_scores)
