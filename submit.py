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
from aoi.video import VideoDetector, read_video_frames
from eval.mvtec import _load_img

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}


def _load_images(d, size):
    return [_load_img(p, size) for p in sorted(Path(d).iterdir()) if p.suffix in IMG_EXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", help="正常样本目录(现场迁移;zero-shot 模式可省)")
    ap.add_argument("--defect", help="缺陷样本目录(现场迁移;zero-shot 模式可省)")
    ap.add_argument("--test", required=True, help="测试目录(图片/视频混合)")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--zeroshot", action="store_true", help="无样本模式:无需 normal/defect,纯 CLIP 判异常")
    ap.add_argument("--class-name", default="object", help="zero-shot 文本提示用的品类名")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.zeroshot:                                  # 无样本入口(赛题"少样本或无样本")
        from aoi.clip_encoder import CLIPEncoder
        from aoi.branches.zeroshot_clip import ZeroShotAdapter
        adapter = ZeroShotAdapter(CLIPEncoder(device=device), class_name=args.class_name)
        adapter.fit_fewshot()
        print(f"模式:zero-shot 无样本(class={args.class_name})", flush=True)
    else:
        if not args.normal or not args.defect:
            raise SystemExit("少样本模式需 --normal 与 --defect;或加 --zeroshot 走无样本")
        bb = Backbone(pretrained=True, device=device)
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
            frames = read_video_frames(str(p), size=args.size)
            out = VideoDetector(adapter).process(frames)
            is_def = len(out["events"]) > 0
            peak = max(out["smoothed_scores"]) if out["smoothed_scores"] else 0.0
            detail = f"events={out['events']}" if is_def else "normal"
            rows.append([p.name, "video", int(is_def), round(peak, 4), detail])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "type", "is_defect", "score", "detail"])
        w.writerows(rows)
    n_def = sum(r[2] for r in rows)
    print(f"处理 {len(rows)} 个样本(缺陷 {n_def});阈值={adapter.threshold:.4f} → {args.out}")


if __name__ == "__main__":
    main()
