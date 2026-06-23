"""提交入口(评委评测用):
  少样本: python submit.py --normal <正常目录> --defect <缺陷目录> --test <测试目录> --out result.csv
  无样本: python submit.py --zeroshot --class-name <品类> --test <测试目录>

对未知产品:用 normal/(~100 正常)+ defect/(~30 缺陷)现场迁移(fit_fewshot,无梯度训练),
再对 test/ 下每个**图片或视频**输出判决。视频走逐帧+时序平滑+事件聚合(早期拦截)。
按图尺寸自动路由:长边≥1024 → 大图混合(全局5分支@320 + 局部ResNet18分块);否则常规 resize。"""
import argparse
import csv
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
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


def _run_zeroshot(args, device):
    """无样本入口(赛题"少样本或无样本"):无需 normal/defect,纯 CLIP 判异常。"""
    from aoi.clip_encoder import CLIPEncoder
    from aoi.branches.zeroshot_clip import ZeroShotAdapter
    adapter = ZeroShotAdapter(CLIPEncoder(device=device), class_name=args.class_name)
    adapter.fit_fewshot()
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


def _run_small(args, bb):
    """常规路径:统一 resize 到 size,跑默认 5 分支 ensemble。"""
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
    """大图(2500²)路径:混合检测器=全局5分支ensemble@320 + 局部ResNet18分块,可靠性软融合。
    全局抓纹理/色彩/尺寸/缺件,局部抓高频小缺陷;延时 GPU<200ms、可 OpenVINO 冲 CPU<2s。"""
    from aoi.hybrid import HybridDetector
    from aoi.video import moving_average, group_events
    global_bb = Backbone(pretrained=True, device=device)                       # 全局 WRN50@320
    local_bb = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=device)
    det = HybridDetector(global_bb, local_bb,
                         local_kw=dict(tile=512, stride=512, coreset_ratio=0.01, feat_grid=32))
    normals = [_load_img_native(p) for p in _img_files(args.normal)]
    defects = [_load_img_native(p) for p in _img_files(args.defect)]
    if not normals or not defects:
        raise SystemExit("normal/ 与 defect/ 必须各含至少一张图片")
    det.fit_fewshot(normals, defects)
    rows = []
    for p in sorted(Path(args.test).iterdir()):
        if p.suffix in IMG_EXT:
            o = det.predict(_load_img_native(p))
            rows.append([p.name, "image", int(o["is_defect"]), round(o["score"], 4),
                         "defect" if o["is_defect"] else "normal"])
        elif p.suffix in VID_EXT:
            frames = read_video_frames(str(p), size=2500)        # 大图视频:逐帧混合
            scores = [det._fused(f) for f in frames]
            sm = moving_average(scores, 3)
            events = group_events([s >= det.threshold for s in sm], 2)
            is_def = len(events) > 0
            rows.append([p.name, "video", int(is_def), round(max(sm) if sm else 0.0, 4),
                         f"events={events}" if is_def else "normal"])
    return rows, det.threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", help="正常样本目录(现场迁移;zero-shot 模式可省)")
    ap.add_argument("--defect", help="缺陷样本目录(现场迁移;zero-shot 模式可省)")
    ap.add_argument("--test", required=True, help="测试目录(图片/视频混合)")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--mode", choices=["auto", "small", "large"], default="auto",
                    help="auto=按图尺寸自动路由;large=强制大图混合;small=常规")
    ap.add_argument("--zeroshot", action="store_true", help="无样本模式:无需 normal/defect,纯 CLIP 判异常")
    ap.add_argument("--class-name", default="object", help="zero-shot 文本提示用的品类名")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.zeroshot:
        print(f"路由:zero-shot 无样本(class={args.class_name})", flush=True)
        rows, thr = _run_zeroshot(args, device)
    else:
        if not args.normal or not args.defect:
            raise SystemExit("少样本模式需 --normal 与 --defect;或加 --zeroshot 走无样本")
        large = args.mode == "large" or (args.mode == "auto" and _decide_large(args.normal))
        if large:
            print("路由:大图混合(全局5分支@320 + 局部ResNet18分块)", flush=True)
            rows, thr = _run_large(args, device)
        else:
            print("路由:常规(resize 5分支ensemble)", flush=True)
            rows, thr = _run_small(args, Backbone(pretrained=True, device=device))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "type", "is_defect", "score", "detail"])
        w.writerows(rows)
    n_def = sum(r[2] for r in rows)
    print(f"处理 {len(rows)} 个样本(缺陷 {n_def});阈值={thr:.4f} → {args.out}")


if __name__ == "__main__":
    main()
