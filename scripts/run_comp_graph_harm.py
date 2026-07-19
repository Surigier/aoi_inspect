"""组件图"默认开"安全性检查:在非逻辑缺陷产品(AD2纹理/外观类)上强制开启组件图,
量它伤不伤纯定位IoU。背景:门控两版(生产base/OOF base)都在fit侧系统性低估边际增益
(juice_bottle估-0.112/-0.037,test真值+0.081)——fit侧测不准正收益,只能反过来问:
"如果默认开,最坏伤多少?"若纹理类≈零伤害(组件=纹理块,统计稳定,z少发火),则可换
"默认开+宽松止损门(gain<-0.05才关)"策略,吃下逻辑缺陷的大额收益。
用法:PYTHONPATH=. python scripts/run_comp_graph_harm.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.component_graph import ComponentGraph
from scripts.run_sam_gate_ab import prep_ad2
from scripts.run_comp_graph_ab import eval_mode

CATS = ["sheet_metal", "walnuts", "fruit_jelly"]


def main():
    torch.manual_seed(0)
    rows = []
    for cat in CATS:
        normals, fit_i, fit_m, test_defs = prep_ad2(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        cg = ComponentGraph(device=det._bb_loc.device)
        cg.fit(det, normals, defect_imgs=fit_i, defect_masks=fit_m)
        n_comp = len(cg.comp_masks) if cg.comp_masks else 0
        acc_b, g_b, p_b, h_b, l_b = eval_mode(det, test_defs, [], None)
        if n_comp >= 2:
            acc_c, g_c, p_c, h_c, l_c = eval_mode(det, test_defs, [], cg)
        else:
            acc_c, g_c, p_c, h_c, l_c = acc_b, g_b, p_b, h_b, l_b
        rows.append((p_b, p_c))
        print(f"{cat:14s} 组件数={n_comp} 门控gain={cg.gain if cg.gain is not None else float('nan'):+.3f} | "
              f"base纯定位={p_b:.3f} 强制+cg={p_c:.3f} Δ={p_c-p_b:+.3f} | lat {l_b:.0f}->{l_c:.0f}ms", flush=True)
    b = np.mean([r[0] for r in rows]); c = np.mean([r[1] for r in rows])
    print(f"\n=== 均值 === base={b:.3f} +cg={c:.3f} Δ={c-b:+.3f}(默认开策略要求≈0)", flush=True)


if __name__ == "__main__":
    main()
