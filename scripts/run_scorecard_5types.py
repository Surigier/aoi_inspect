"""严格按赛题5类缺陷(尺寸偏差、缺件少件、逻辑错误、色彩变化、常见外观缺陷)跑一遍
真口径成绩单(acc/含漏检IoU/框命中@0.5/延时,无AUROC)。

已有数据直接复用production scorecard的口径(不重复造轮子):
  常见外观缺陷 → hazelnut(crack/cut/hole)
  缺件少件     → cable(missing_cable/missing_wire) + pcb的QS(缺件)子集(新)
  色彩变化     → pill(color) + carpet/leather/metal_nut/wood的color子目录(新)
  逻辑错误(如顺序错误) → LOCO logical_anomalies(breakfast_box/splicing_connectors/
                screw_bag,⚠️LOCO官方定义是"计数/位置/组合错误",不完全等同赛题字面
                "顺序错误",邮件已列为待问出题人的问题)
  尺寸偏差     → 本地7个数据集(MVTec/LOCO/Real-IAD/DAGM/VisA/MPDD/pku_pcb)+外部
                搜索均未找到明确标注"尺寸偏差"的开源数据,如实标"无数据"

用法:PYTHONPATH=. python scripts/run_scorecard_5types.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import gt_boxes, box_hit, _read, evaluate as evaluate_mvtec_style

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256)
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")


def prep_mvtec_color(cat):
    root = Path(f"data/mvtec/{cat}")
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    df = [(p, "color") for p in sorted(glob.glob(str(root / "test/color/*.png")))]
    random.Random(0).shuffle(df); k = max(3, len(df) // 3)
    fit_i = [load_fast(p) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth/color" / (Path(p).stem + "_mask.png"), HW) for p, _ in df[:k]]
    test_defs = [(load_fast(p), _read(GT / cat / "ground_truth/color" / (Path(p).stem + "_mask.png"), HW))
                for p, _ in df[k:]]
    goods = [(load_fast(p), None) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    return normals, fit_i, fit_m, test_defs, goods


def prep_realiad_class(cat, target_class, n_norm=100, n_fit=30, n_test=40, seed=0):
    """按anomaly_class筛出指定缺陷类型(如QS=缺件少件),同production口径fit/test切分。"""
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(seed).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:n_norm]]
    ng = [x for x in d["test"] if x["anomaly_class"] == target_class]
    random.Random(seed).shuffle(ng)
    fit_i = [load_fast(R / x["image_path"]) for x in ng[:n_fit]]
    fit_m = [_read(R / x["mask_path"], HW) for x in ng[:n_fit]]
    test_defs = [(load_fast(R / x["image_path"]), _read(R / x["mask_path"], HW))
                for x in ng[n_fit:n_fit + n_test]]
    goods = [(load_fast(R / x["image_path"]), None) for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== 严格按赛题5类缺陷的真口径成绩单(acc/含漏检IoU/框命中@0.5/延时) ===\n", flush=True)

    print("--- 色彩变化:扩展到carpet/leather/metal_nut/wood(此前只有pill) ---", flush=True)
    color_rows = []
    for cat in ["carpet", "leather", "metal_nut", "wood"]:
        r = evaluate_mvtec_style(f"色彩(新){cat}", *prep_mvtec_color(cat))
        color_rows.append(r)

    print("\n--- 缺件少件:pcb的QS(缺件少件)子集,104张真实标注,此前只有cable ---", flush=True)
    qs_row = evaluate_mvtec_style("缺件(新)pcb-QS", *prep_realiad_class("pcb", "QS"))

    print("\n=== 汇总:色彩变化(pill+4类新增) ===", flush=True)
    all_color = color_rows  # pill的0.440已在原scorecard报过,这里汇总新增4类
    a = np.mean([r[0] for r in all_color]); g = np.mean([r[1] for r in all_color])
    h = np.mean([r[3] for r in all_color])
    print(f"新增4类均值: acc={a:.3f} IoU={g:.3f} 框命中={h:.3f}(pill单独: acc=1.000 IoU=0.440 框=0.426,已有数据)", flush=True)

    print("\n=== 缺件少件:cable(0.927/0.811/0.933,已有) vs pcb-QS(新) ===", flush=True)
    print(f"pcb-QS: acc={qs_row[0]:.3f} IoU={qs_row[1]:.3f} 框命中={qs_row[3]:.3f}", flush=True)

    print("\n=== 尺寸偏差:本地7个数据集+外部搜索均无标注支撑,无数据可报 ===", flush=True)


if __name__ == "__main__":
    main()
