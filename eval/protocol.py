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


def run_protocol(adapter, normals_fit, defects_fit, test_images, test_labels):
    """复刻官方协议:fit_fewshot 迁移后在测试集上算 AUROC/准确率/平均延时。"""
    adapter.fit_fewshot(normals_fit, defects_fit)
    scores, lats, preds = [], [], []
    for img in test_images:
        r, is_def = adapter.predict(img.unsqueeze(0))
        scores.append(r.score)
        lats.append(r.latency_ms)
        preds.append(int(is_def))
    scores = np.array(scores)
    labels = np.array(test_labels)
    preds = np.array(preds)
    return {
        "auroc": image_auroc(scores, labels),
        "accuracy": float((preds == labels).mean()),
        "latency_ms_mean": float(np.mean(lats)),
    }
