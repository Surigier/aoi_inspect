"""实测验证:操作员反馈是否**真的动态改变了模型参数**,以及漏检是否被救回。

赛题原文要求"动态调整模型参数"。仅靠读代码说"会重训"是不够的,这里做端到端实测:
  1) 用29张缺陷fit(留1张不给),在测试集里找一张**被漏检**的图
  2) 记录:监督分割头的权重指纹、判决阈值、DINO门阈值、像素阈值
  3) 操作员反馈那张漏检图(带掩膜)
  4) 复查:权重是否真的变了、那张图是否被救回、其余样本有没有被带崩

用法:PYTHONPATH=. python scripts/verify_feedback.py [类目=cable]
"""
import glob
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aoi.competition import CompetitionLargeDetector
from aoi.active_learning import ActiveLearningLoop

GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def mask_of(p, hw=(256, 256)):
    return (np.array(Image.open(p).convert("L").resize(hw[::-1], Image.NEAREST)) > 0).astype(np.uint8)


def fingerprint(det):
    """分割头权重的指纹 + 各个标定量,用来判断反馈到底改了什么。"""
    h = "无头"
    if det.seg_head.head is not None:
        b = b"".join(p.detach().cpu().numpy().tobytes() for p in det.seg_head.head.parameters())
        h = hashlib.md5(b).hexdigest()[:12]
    return {
        "分割头权重指纹": h,
        "判决阈值": round(float(det.threshold), 6) if det.threshold is not None else None,
        "DINO门阈值": round(float(det._dino_thr), 6) if getattr(det, "_dino_thr", None) is not None else None,
        "像素阈值": round(float(det.pix_thr), 6) if det.pix_thr is not None else None,
        "类型头就绪": bool(getattr(det.type_head, "ready", False)),
    }


def main(cat="cable"):
    torch.manual_seed(0)
    root = Path(f"data/mvtec/{cat}")
    ns = sorted(glob.glob(str(root / "train/good/*.png")))[:100]
    df = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            for f in sorted(sub.glob("*.png")):
                m = GT / cat / "ground_truth" / sub.name / (f.stem + "_mask.png")
                if m.exists():
                    df.append((str(f), str(m)))
    random.Random(0).shuffle(df)
    fit_d, test_d = df[:29], df[29:60]

    print(f"[1/4] 初始迁移学习:{cat} 正常{len(ns)} + 缺陷{len(fit_d)}", flush=True)
    det = CompetitionLargeDetector()
    loop = ActiveLearningLoop(det, [load(p) for p in ns],
                              [load(p) for p, _ in fit_d],
                              defect_masks=[mask_of(m) for _, m in fit_d])
    before = fingerprint(det)
    print(f"      反馈前:{before}", flush=True)

    print("[2/4] 在测试集里找一张被漏检的缺陷图", flush=True)
    missed = None
    for p, m in test_d:
        img = load(p)
        if not det.locate(img)["is_defect"]:
            missed = (p, m, img); break
    if missed is None:
        print("      本轮没有漏检样本(检出率100%),改用最低分的缺陷图做反馈演示", flush=True)
        p, m = min(test_d, key=lambda x: det.locate(load(x[0]))["score"])
        missed = (p, m, load(p))
    print(f"      漏检样本:{Path(missed[0]).parent.name}/{Path(missed[0]).name}", flush=True)

    print("[3/4] 操作员反馈该漏检样本(带掩膜)", flush=True)
    t0 = time.time()
    n_norm, n_def = loop.feedback(missed[2], is_defect=True, mask=mask_of(missed[1]))
    sec = time.time() - t0
    diag = getattr(loop, "last_diagnosis", None)
    print(f"      耗时 {sec:.0f}s  样本库 正常{n_norm}/缺陷{n_def}", flush=True)
    print(f"      VLM即时诊断:{diag}", flush=True)

    print("[4/4] 复查", flush=True)
    after = fingerprint(det)
    print(f"      反馈后:{after}", flush=True)
    changed = [k for k in before if before[k] != after[k]]
    print(f"\n=== 参数是否真的变了 ===", flush=True)
    for k in before:
        mark = "✅变了" if before[k] != after[k] else "— 未变"
        print(f"  {k:14s} {mark}   {before[k]} → {after[k]}", flush=True)
    o = det.locate(missed[2])
    print(f"\n=== 该漏检样本是否被救回 ===", flush=True)
    print(f"  反馈后判定:{'✅ 检出' if o['is_defect'] else '❌ 仍漏检'} "
          f"(异常分{o['score']:.4f} / 阈值{det.decision_threshold():.4f})", flush=True)
    rest = [x for x in test_d if x[0] != missed[0]][:20]
    det_n = sum(1 for p, _ in rest if det.locate(load(p))["is_defect"])
    print(f"\n=== 其余{len(rest)}张缺陷有没有被带崩 ===", flush=True)
    print(f"  反馈后检出 {det_n}/{len(rest)} = {det_n/max(len(rest),1):.1%}", flush=True)
    print("VERIFY_FEEDBACK OK", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cable")
