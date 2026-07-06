"""极简诊断:breakfast_box 上 single/tmpl_diff/dino/dino_cat 四候选的留出IoU(不跑EAD/DINO门)。
四候选 extractor 均不依赖 EAD → 跳过 fit_fewshot,直接建器对比。
用法:PYTHONPATH=. python scripts/diag_featsel.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector, _mask_np
from aoi.seg_head import SupervisedSegHead
from aoi.tmpl_ref import RefBank
from scripts.run_logic_scorecard import prep_logic

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test, _ = prep_logic("breakfast_box")
    det = CompetitionLargeDetector(sam_refine=False)          # 只需 extractor,不建SAM
    det._ref_bank = RefBank(normals)
    print("extractors ready", flush=True)
    hold = list(range(0, len(fit_i), 4)); tr = [i for i in range(len(fit_i)) if i not in set(hold)]

    def _try(name, ex):
        try:
            h = SupervisedSegHead(device=det.seg_head.device, steps=150, extractor=ex)
            ok = h.fit(None, [fit_i[i] for i in tr], [fit_m[i] for i in tr], normals[:15])
            if not ok or h.thr is None:
                return "no-fit"
            ious = []
            for i in hold:
                amap = h.map(None, fit_i[i], det.seg_eval_hw)
                gt = _mask_np(fit_m[i], det.seg_eval_hw)
                pred = amap >= h.thr
                TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return round(float(np.mean(ious)), 3)
        except Exception as e:
            return f"ERR:{type(e).__name__}:{e}"

    for name, ex in [("single", det._wrn_feats), ("tmpl_diff", det._wrn_feats_diff),
                     ("dino", det._dino_feats), ("dino_cat", det._dino_cat_feats)]:
        print(f"  {name:10s} 留出IoU = {_try(name, ex)}", flush=True)


if __name__ == "__main__":
    main()
