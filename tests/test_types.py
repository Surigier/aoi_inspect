import numpy as np
from aoi.types import BranchResult

def test_branch_result_defaults():
    r = BranchResult(score=0.7)
    assert r.score == 0.7
    assert r.anomaly_map is None
    assert r.boxes == []
    assert r.defect_type == "unknown"
    assert r.latency_ms == 0.0

def test_branch_result_with_map():
    m = np.zeros((4, 4))
    r = BranchResult(score=1.0, anomaly_map=m, defect_type="appearance", latency_ms=12.3)
    assert r.anomaly_map.shape == (4, 4)
    assert r.defect_type == "appearance"
