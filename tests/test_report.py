import numpy as np
from aoi.report import summarize_detection, format_report, DefectReporter
from aoi.types import BranchResult


def test_summarize_extracts_peak_cell():
    amap = np.zeros((4, 4)); amap[2, 3] = 9.0
    r = BranchResult(score=1.5, anomaly_map=amap, defect_type="structural")
    facts = summarize_detection(r, is_defect=True)
    assert facts["is_defect"] is True
    assert facts["defect_type"] == "structural"
    assert facts["peak_cell"] == (2, 3)
    assert facts["grid"] == (4, 4)


def test_format_report_normal_vs_defect():
    normal = format_report({"is_defect": False, "defect_type": "normal", "score": 0.1})
    assert "正常" in normal
    defect = format_report({"is_defect": True, "defect_type": "appearance", "score": 2.0,
                            "peak_cell": (1, 1), "grid": (8, 8)})
    assert "缺陷" in defect and "appearance" in defect


def test_reporter_template_and_llm():
    r = BranchResult(score=2.0, defect_type="appearance")
    txt = DefectReporter().report(r, is_defect=True)
    assert "缺陷" in txt
    captured = {}
    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "LLM报告"
    out = DefectReporter(llm_fn=fake_llm).report(r, is_defect=True)
    assert out == "LLM报告"
    assert "appearance" in captured["prompt"]
