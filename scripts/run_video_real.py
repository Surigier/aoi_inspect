"""视频真实化验证(官方:视频=测试图合成)。
用真实测试图序列拼成视频流:正常帧→真实缺陷帧段→正常帧,跑逐帧VideoDetector,
报检出的缺陷事件、早期拦截延迟、正常段误报。比合成缺陷更贴官方定义。
用法:python scripts/run_video_real.py
"""
import glob
import random
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.video import VideoDetector
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 320


def build_stream(cat, defect_folder, n_pre=10, n_def=8, n_post=10):
    """真实帧序列:正常→缺陷段(真实缺陷图)→正常。返回 (frames, defect_frame_indices)。"""
    root = Path(f"data/mvtec/{cat}")
    goods = sorted(glob.glob(str(root / "test/good/*.png")))
    defs = sorted(glob.glob(str(root / "test" / defect_folder / "*.png")))
    random.Random(0).shuffle(goods); random.Random(1).shuffle(defs)
    seq = goods[:n_pre] + defs[:n_def] + goods[n_pre:n_pre + n_post]
    frames = [_load_img(p, SIZE) for p in seq]
    def_idx = list(range(n_pre, n_pre + n_def))
    return frames, def_idx


def main():
    torch.manual_seed(0)
    cat, folder = "cable", "missing_cable"
    root = Path(f"data/mvtec/{cat}")
    bb = Backbone(pretrained=True, device=DEV)
    adapter = default_adapter(bb)
    normals = [_load_img(p, SIZE) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    dfiles = sorted(glob.glob(str(root / "test" / folder / "*.png")))
    defects = [_load_img(p, SIZE) for p in dfiles[:15]]
    adapter.fit_fewshot(normals, defects)

    frames, def_idx = build_stream(cat, folder)
    out = VideoDetector(adapter, smooth_window=3, min_event_len=2).process(frames)
    events = out["events"]; sm = out["smoothed_scores"]; thr = adapter.threshold

    print(f"=== 真实视频流({cat}, {len(frames)}帧: 正常→{folder}缺陷段[帧{def_idx[0]}-{def_idx[-1]}]→正常)===")
    print(f"阈值={thr:.3f}")
    print(f"检出事件(帧区间): {events}")
    # 评估:缺陷段是否被检出 + 早期拦截 + 正常段误报
    def_set = set(def_idx)
    hit = any(not def_set.isdisjoint(range(s, e + 1)) for s, e in events)
    fired = [i for i, s in enumerate(sm) if s >= thr]
    first_fire = next((i for i in fired if i in def_set), None)
    delay = (first_fire - def_idx[0]) if first_fire is not None else None
    fp = [i for i in fired if i not in def_set]
    print(f"缺陷段检出: {'✅是' if hit else '❌否'}")
    print(f"首次触发帧: {first_fire}(缺陷段起始{def_idx[0]}) → 早期拦截延迟: {delay}帧")
    print(f"正常段误报帧: {fp if fp else '无'}")
    print(f"逐帧平滑分(缺陷段加粗): " + " ".join(
        (f"[{s:.2f}]" if i in def_set else f"{s:.2f}") for i, s in enumerate(sm)))


if __name__ == "__main__":
    main()
