import numpy as np


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """秩和法计算 image-level AUROC。labels: 1=缺陷, 0=正常。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
