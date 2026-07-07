"""验证大图视频路径吃到DINO门(复现submit.py _run_large视频逻辑,用frame_score/decision_threshold)。
真实帧流:正常→缺陷段→正常,查:DINO门是否启用、缺陷段检出、正常段误报。
用法:PYTHONPATH=. python scripts/verify_video_gate.py
"""
import glob
import random
from pathlib import Path
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.video import moving_average, group_events
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    cat, folder = "cable", "missing_cable"
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    dfiles = sorted(glob.glob(str(root / "test" / folder / "*.png")))
    defects = [_load_img(p, 640) for p in dfiles[:20]]
    det = CompetitionLargeDetector(device=DEV, sam_refine=False)
    det.fit_fewshot(normals, defects[:15], defect_masks=None)

    goods = sorted(glob.glob(str(root / "test/good/*.png")))
    random.Random(0).shuffle(goods)
    n_pre = n_post = 10; n_def = 8
    seq = goods[:n_pre] + dfiles[15:15 + n_def] + goods[n_pre:n_pre + n_post]
    frames = [_load_img(p, 640) for p in seq]
    def_idx = set(range(n_pre, n_pre + n_def))

    # 新路径(融合门) vs 旧路径(EAD分+EAD阈值),同批帧对比
    ead_raw = [det.branches[0].score(f) for f in frames]
    fused = [det.frame_score(f) for f in frames]
    thr_f = det.decision_threshold(); thr_e = det.threshold
    print(f"=== 大图视频门验证({cat}/{folder}, {len(frames)}帧, 缺陷帧{sorted(def_idx)})===", flush=True)
    print(f"DINO门启用={det._dino is not None} | 融合阈值={thr_f:.3f} | EAD阈值={thr_e:.3f}", flush=True)

    def report(tag, scores, thr):
        sm = moving_average(scores, 3)
        events = group_events([thr is not None and s >= thr for s in sm], 2)
        fired = [i for i, s in enumerate(sm) if s >= thr]
        hit = any(not def_idx.isdisjoint(range(s, e + 1)) for s, e in events)
        fp = [i for i in fired if i not in def_idx]
        print(f"[{tag}] 事件={events} 缺陷检出={'✅' if hit else '❌'} 误报={fp if fp else '无'}", flush=True)
        print(f"      逐帧平滑(缺陷段[]): " + " ".join(
            (f"[{s:.2f}]" if i in def_idx else f"{s:.2f}") for i, s in enumerate(sm)), flush=True)

    report("新-融合门", fused, thr_f)
    report("旧-EAD门", ead_raw, thr_e)
    print(f"fit融合分范围: 正常 见标定 | 缺陷段原始融合分: "
          f"{[round(fused[i],2) for i in sorted(def_idx)]}", flush=True)


if __name__ == "__main__":
    main()
