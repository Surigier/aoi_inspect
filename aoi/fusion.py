def znorm(score: float, mean: float, std: float) -> float:
    """按正常分的均值/方差做 z 归一化;std≈0 时退化为减均值,避免除零。"""
    return (score - mean) / (std if std > 1e-12 else 1.0)


def fuse(norm_scores) -> float:
    """多分支归一化分数融合:取最大(任一分支报异常即视为异常)。"""
    return max(norm_scores)


def auroc(scores, labels) -> float:
    """秩和法 image-AUROC(纯 Python,无 numpy 依赖,用于估分支可分性)。
    labels: 1=缺陷, 0=正常。两类有缺则返回 0.5。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    n = len(order)
    i = 0
    while i < n:                                  # 同分赋平均秩
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [ranks[i] for i in range(len(labels)) if labels[i] == 1]
    n_pos = len(pos)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
