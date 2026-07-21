"""cable确定性复现(acc=0.255稳定,两次独立进程完全一致)后的标定归因:
融合阈值fit时标在30张缺陷+100张正常上,test时套到15张缺陷+40张正常——打印fit侧
和test侧的融合分分布,看阈值相对两边的位置,定位标定为什么系统性偏低(正常图误判)。
用法:PYTHONPATH=. python scripts/diag_cable_threshold.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_mvtec


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs, goods = prep_mvtec("cable", ["missing_cable", "missing_wire"])
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"DINO门={'开' if det._dino is not None else '关'}  阈值={det._dino_thr:.4f}", flush=True)

    def fused(img):
        return det._dino_fuse(det.branches[0].score(img), det._dino.score(img))

    fit_n_scores = [fused(n) for n in normals]              # fit标定用的100张正常
    fit_d_scores = [fused(d) for d in fit_i]                # fit标定用的30张缺陷
    test_n_scores = [fused(g) for g, _ in goods]             # test 40张正常(未参与标定)
    test_d_scores = [fused(d) for d, _ in test_defs]         # test剩余缺陷(未参与标定)

    thr = det._dino_thr
    print(f"fit正常  : 中位={np.median(fit_n_scores):.3f} p90={np.percentile(fit_n_scores,90):.3f} "
          f"max={max(fit_n_scores):.3f}", flush=True)
    print(f"fit缺陷  : 中位={np.median(fit_d_scores):.3f} p10={np.percentile(fit_d_scores,10):.3f} "
          f"min={min(fit_d_scores):.3f}", flush=True)
    print(f"test正常 : 中位={np.median(test_n_scores):.3f} p90={np.percentile(test_n_scores,90):.3f} "
          f"max={max(test_n_scores):.3f}  超阈值(误报)={sum(s>=thr for s in test_n_scores)}/{len(test_n_scores)}", flush=True)
    print(f"test缺陷 : 中位={np.median(test_d_scores):.3f} p10={np.percentile(test_d_scores,10):.3f} "
          f"min={min(test_d_scores):.3f}  超阈值(命中)={sum(s>=thr for s in test_d_scores)}/{len(test_d_scores)}", flush=True)
    print(f"阈值={thr:.3f}", flush=True)

    # EAD/DINO两支各自的贡献:哪一支在test正常图上把分数顶过了阈值?
    emu, esd, dmu, dsd = det._dino_stats
    for g, _ in goods[:8]:
        ez = (det.branches[0].score(g) - emu) / esd
        dz = (det._dino.score(g) - dmu) / dsd
        flag = "!!误报" if max(ez, dz) >= thr else ""
        print(f"  test正常样例: z_EAD={ez:.3f} z_DINO={dz:.3f} fused={max(ez,dz):.3f} {flag}", flush=True)


if __name__ == "__main__":
    main()
