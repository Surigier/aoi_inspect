"""诊断SAM门控在pcb/battery上的真实决策(排查是否误判关闭SAM,还是训练方差)。
快速跑单类fit,打印 on/off IoU 均值 + 最终门控开关状态,不需要跑完整成绩单。
用法:PYTHONPATH=. python scripts/diag_sam_gate.py [类名...]
"""
import sys
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad, prep_mvtec


def diag(name, prep):
    normals, fit_i, fit_m, test_defs, test_goods = prep()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    on, off = getattr(det, "sam_gate_debug", (None, None))
    gated_off = getattr(det, "sam_gated_off", False)
    sam_final = det.sam is not None
    print(f"{name:12s} SAM-on IoU={on} SAM-off IoU={off}  gated_off={gated_off}  最终SAM启用={sam_final}", flush=True)


def main():
    jobs = {
        "pcb": lambda: prep_realiad("pcb"),
        "battery": lambda: prep_realiad("phone_battery"),
        "hazelnut": lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]),
        "cable": lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"]),
    }
    for name in (sys.argv[1:] or ["pcb", "battery"]):
        diag(name, jobs[name])


if __name__ == "__main__":
    main()
