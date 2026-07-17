"""优先级2关键补验:SAM逐区域OOF门控在【SAM历史正增益域】(Real-IAD pcb/phone_battery,
项目记录+23%~50%)上的行为——AD2三类验证只证明了"消除伤害"(全判reject_all),还没证明
门控不会把该保留的收益一刀切拒掉。此处若gate学出规则(非reject_all)且new_sam≥old_sam≈
保留历史增益,则门控完整闭环;若错判reject_all导致丢增益,则需要调margin。
三路对比同run_sam_gate_ab.py:raw(无SAM)/old(总接受)/new(OOF门控)。
用法:PYTHONPATH=. python scripts/run_sam_gate_realiad.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad
from scripts.run_sam_gate_ab import eval_pure_iou


def main():
    torch.manual_seed(0)
    cats = ["pcb", "phone_battery"]
    results = {}
    for cat in cats:
        normals, fit_i, fit_m, test_defs, _goods = prep_realiad(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        gate = getattr(det.sam, "gate", None) if det.sam else None
        gain = getattr(det.sam, "calib_gain", None) if det.sam else None
        pad = getattr(det.sam, "padding", None) if det.sam else None
        raw_iou = eval_pure_iou(det, test_defs, "raw")
        old_iou = eval_pure_iou(det, test_defs, "old")
        new_iou = eval_pure_iou(det, test_defs, "new")
        results[cat] = (raw_iou, old_iou, new_iou)
        print(f"{cat:14s} raw(无SAM)={raw_iou:.3f}  old_sam(总接受)={old_iou:.3f}  "
              f"new_sam(OOF门控)={new_iou:.3f}  gate={gate}  OOF gain={gain}  padding={pad}", flush=True)
    print("\n=== 均值 ===")
    r = np.mean([v[0] for v in results.values()])
    o = np.mean([v[1] for v in results.values()])
    n = np.mean([v[2] for v in results.values()])
    print(f"raw={r:.3f}  old_sam={o:.3f}  new_sam={n:.3f}  "
          f"Δ(new-raw)={n-r:+.3f}  Δ(new-old)={n-o:+.3f}", flush=True)
    print("判读:old>raw(SAM这里真有用)时,new≈old为门控保住收益;new≈raw<old为错杀(需调margin)。", flush=True)


if __name__ == "__main__":
    main()
