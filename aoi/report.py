import numpy as np


def summarize_detection(result, is_defect) -> dict:
    """从 BranchResult 抽结构化事实。"""
    facts = {
        "is_defect": bool(is_defect),
        "defect_type": result.defect_type,
        "score": float(result.score),
    }
    if result.anomaly_map is not None:
        a = np.asarray(result.anomaly_map)
        idx = np.unravel_index(int(a.argmax()), a.shape)
        facts["peak_cell"] = (int(idx[0]), int(idx[1]))
        facts["grid"] = (int(a.shape[0]), int(a.shape[1]))
    return facts


def format_report(facts: dict) -> str:
    """结构化事实 -> 中文缺陷报告文本。"""
    if not facts["is_defect"]:
        return f"检测结果:正常(异常分 {facts['score']:.3f})。"
    loc = ""
    if "peak_cell" in facts:
        r, c = facts["peak_cell"]
        gh, gw = facts["grid"]
        loc = f",最可疑区域位于 {gh}×{gw} 网格第 ({r},{c}) 格"
    return (f"检测结果:缺陷(类型:{facts['defect_type']},异常分 {facts['score']:.3f}){loc}。"
            "建议人工复核该区域。")


class DefectReporter:
    """冷路径缺陷报告:默认模板;可注入 llm_fn(prompt)->str 生成更自然报告。"""

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def report(self, result, is_defect) -> str:
        facts = summarize_detection(result, is_defect)
        if self.llm_fn is None:
            return format_report(facts)
        prompt = f"根据以下工业质检结果写一段简洁的中文缺陷报告:{facts}"
        return self.llm_fn(prompt)
