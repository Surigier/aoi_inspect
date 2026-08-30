"""Real-IAD 全12类目成绩单 —— 补齐"手机/电子部件域"的真实覆盖面。

为什么做这个:现有权威成绩单(run_scorecard.py)只有5个类目,其中真正对得上赛题
隐藏域"手机部件"的只有2个(pcb、phone_battery)。而 data/_dl/Real-IAD 本地就躺着
**12个类目、每类约2400张OK图、真实像素掩膜**,`prep_realiad(cat)` 是全参数化的,
另外10个直接能跑。跑完才有底气说"覆盖12种电子/手机部件"——这直接喂给评分里
占比最大的"方案完整度"(竞赛得分50%)。

口径与 run_scorecard.py **完全一致**(直接import它的evaluate/prep_realiad,不另写
一套):100正常 + 30缺陷fit,40张缺陷 + 40张正常测,报图级acc/含漏检IoU/纯定位IoU/
框命中@0.5/延时。**无AUROC。**

每跑完一类立刻打印并落盘,中途断了也保住已完成的部分(全跑一遍约4小时)。

用法:PYTHONPATH=. python scripts/run_scorecard_realiad12.py [类目1 类目2 ...]
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scripts.run_scorecard import evaluate, prep_realiad

# 手机部件相关的排在前面——万一跑不完,先拿到最相关的
CATS = ["phone_battery", "sim_card_set", "pcb",           # 赛题隐藏域直接相关
        "usb", "usb_adaptor", "audiojack",                 # 接口类
        "button_battery", "switch", "terminalblock",
        "transistor1", "regulator", "end_cap"]
OUT = Path("_logs/scorecard_realiad12.json")


def main(cats, fresh=False):
    """fresh=True 强制重算,忽略续跑缓存。

    **续跑缓存必须能被强制绕过**:它曾在一次"验证修复是否生效"的运行里静默重放了
    修复前的陈旧结果("已有结果,跳过"),导致把"修复无效"当成结论、并据此白查了
    两轮不存在的"运行间不确定性"。缓存对续跑是好事,对验证是陷阱。"""
    torch.manual_seed(0)
    print(f"=== Real-IAD {len(cats)}类目成绩单(竞赛口径,无AUROC)==="
          + ("  [--fresh 强制重算]" if fresh else ""), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    done = {} if fresh else (json.loads(OUT.read_text()) if OUT.exists() else {})
    for c in cats:
        if c in done:
            print(f"{c:18s} 已有结果,跳过", flush=True)
            continue
        t0 = time.time()
        try:
            acc, iou_g, iou_p, hit = evaluate(c, *prep_realiad(c))
        except Exception as e:
            print(f"{c:18s} 失败: {type(e).__name__}: {e}", flush=True)
            continue
        done[c] = dict(acc=float(acc), iou_gated=float(iou_g),
                       iou_pure=float(iou_p), hit=float(hit), sec=round(time.time() - t0))
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1))   # 每类落盘,断了不丢
    if done:
        k = list(done)
        print(f"\n=== 汇总(n={len(k)}) ===", flush=True)
        for c in k:
            d = done[c]
            print(f"  {c:18s} acc={d['acc']:.3f}  含漏检IoU={d['iou_gated']:.3f}  "
                  f"框命中={d['hit']:.3f}  ({d['sec']}s)", flush=True)
        for f, label in [("acc", "图级acc"), ("iou_gated", "含漏检IoU"),
                         ("iou_pure", "纯定位IoU"), ("hit", "框命中@0.5")]:
            v = [done[c][f] for c in k]
            print(f"{label:12s} 均值={np.mean(v):.3f}  最低={np.min(v):.3f}({k[int(np.argmin(v))]})", flush=True)
    print("REALIAD12 OK", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fresh"]
    main(args or CATS, fresh="--fresh" in sys.argv)
