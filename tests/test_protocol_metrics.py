import numpy as np
from eval.protocol import image_auroc

def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert abs(image_auroc(scores, labels) - 1.0) < 1e-9

def test_auroc_random_is_half():
    # 3 concordant, 3 discordant pairs → AUROC = 0.5
    # pos scores: 0.3, 0.7; neg scores: 0.1, 0.5, 0.9
    scores = np.array([0.1, 0.5, 0.3, 0.7, 0.9])
    labels = np.array([0,   0,   1,   1,   0  ])
    assert abs(image_auroc(scores, labels) - 0.5) < 1e-9
