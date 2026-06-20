"""提交入口(评委评测用):
  python submit.py --normal <正常图目录> --defect <缺陷图目录> --test <测试目录> --out result.csv [--class-name X]

对未知产品:用 normal/(~100 正常)+ defect/(~30 缺陷)现场迁移(fit_fewshot,无梯度训练),
再对 test/ 下每个**图片或视频**输出判决。视频走逐帧+时序平滑+事件聚合(早期拦截)。"""
import argparse
import csv
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.multibranch import MultiBranchAdapter
from aoi.video import VideoDetector, read_video_frames
from eval.mvtec import _load_img

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}


def _load_images(d, size):
    return [_load_img(p, size) for p in sorted(Path(d).iterdir()) if p.suffix in IMG_EXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="正常样本目录(现场迁移)")
    ap.add_argument("--defect", required=True, help="缺陷样本目录(现场迁移)")
    ap.add_argument("--test", required=True, help="测试目录(图片/视频混合)")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(pretrained=True, device=device)
    adapter = MultiBranchAdapter([
        TextureADBranch(backbone=bb),
        StructuralADBranch(backbone=bb, grid_size=16),
    ])
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
