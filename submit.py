"""提交入口(评委评测用):
  python submit.py --normal <正常图目录> --defect <缺陷图目录> --test <测试目录> --out result.csv [--class-name X]

对未知产品:用 normal/(~100 正常)+ defect/(~30 缺陷)现场迁移(fit_fewshot,无梯度训练),
再对 test/ 下每个**图片或视频**输出判决。视频走逐帧+时序平滑+事件聚合(早期拦截)。"""
import argparse
import csv
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.tiled import TiledFewShotDetector
from aoi.video import VideoDetector, read_video_frames
from eval.mvtec import _load_img, _load_img_native, peek_size

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}
LARGE_THRESHOLD = 1024          # 长边 ≥ 此值 → 走大图分块路径(原生分辨率,保小缺陷)


def _img_files(d):
    return [p for p in sorted(Path(d).iterdir()) if p.suffix in IMG_EXT]


def _load_images(d, size):
    return [_load_img(p, size) for p in _img_files(d)]


def _decide_large(normal_dir):
    """看首张正常图长边,≥阈值则走大图分块路径。"""
    files = _img_files(normal_dir)
    return bool(files) and peek_size(files[0]) >= LARGE_THRESHOLD


def _run_small(args, bb):
    """常规路径:统一 resize 到 size,跑默认 ensemble。"""
    adapter = default_adapter(bb)
    normals = _load_images(args.normal, args.size)
    defects = _load_images(args.defect, args.size)
    if not normals or not defects:
        raise SystemExit("normal/ 与 defect/ 必须各含至少一张图片")
    adapter.fit_fewshot(normals, defects)
    rows = []
    for p in sorted(Path(args.test).iterdir()):
        if p.suffix in IMG_EXT:
            r, is_def = adapter.predict(_load_img(p, args.size).unsqueeze(0))
            rows.append([p.name, "image", int(is_def), round(r.score, 4),
                         r.defect_type if is_def else "normal"])
        elif p.suffix in VID_EXT:
            out = VideoDetector(adapter).process(read_video_frames(str(p), size=args.size))
            is_def = len(out["events"]) > 0
            peak = max(out["smoothed_scores"]) if out["smoothed_scores"] else 0.0
            rows.append([p.name, "video", int(is_def), round(peak, 4),
                         f"events={out['events']}" if is_def else "normal"])
    return rows, adapter.threshold


def _run_large(args, device):
    """大图分块路径:原生分辨率 + ResNet18 小骨干 + 分块聚合(保小缺陷、控延时)。"""
    from aoi.video import moving_average, group_events
    bb = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=device)
    det = TiledFewShotDetector(bb, tile=512, stride=512, coreset_ratio=0.01, feat_grid=32)
    normals = [_load_img_native(p) for p in _img_files(args.normal)]
    defects = [_load_img_native(p) for p in _img_files(args.defect)]
    if not normals or not defects:
        raise SystemExit("normal/ 与 defect/ 必须各含至少一张图片")
    det.fit_fewshot(normals, defects)
    rows = []
    for p in sorted(Path(args.test).iterdir()):
        if p.suffix in IMG_EXT:
            o = det.predict(_load_img_native(p))
            loc = f"tile={o['worst_tile']}" if o["is_defect"] else "normal"
            rows.append([p.name, "image", int(o["is_defect"]), round(o["score"], 4), loc])
        elif p.suffix in VID_EXT:
            frames = read_video_frames(str(p), size=2500)        # 大图视频:逐帧原生分块
            scores = [det._image_score(f)[0] for f in frames]
            sm = moving_average(scores, 3)
            events = group_events([s >= det.threshold for s in sm], 2)
            is_def = len(events) > 0
            rows.append([p.name, "video", int(is_def), round(max(sm) if sm else 0.0, 4),
                         f"events={events}" if is_def else "normal"])
    return rows, det.threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="正常样本目录(现场迁移)")
    ap.add_argument("--defect", required=True, help="缺陷样本目录(现场迁移)")
    ap.add_argument("--test", required=True, help="测试目录(图片/视频混合)")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--mode", choices=["auto", "small", "large"], default="auto",
                    help="auto=按图尺寸自动路由;large=强制大图分块;small=常规")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    large = args.mode == "large" or (args.mode == "auto" and _decide_large(args.normal))
    if large:
        print("路由:大图分块(原生分辨率+ResNet18)", flush=True)
        rows, thr = _run_large(args, device)
    else:
        print("路由:常规(resize)", flush=True)
        rows, thr = _run_small(args, Backbone(pretrained=True, device=device))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "type", "is_defect", "score", "detail"])
        w.writerows(rows)
    n_def = sum(r[2] for r in rows)
    print(f"处理 {len(rows)} 个样本(缺陷 {n_def});阈值={thr:.4f} → {args.out}")


if __name__ == "__main__":
    main()
