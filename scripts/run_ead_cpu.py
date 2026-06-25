"""EfficientAD 大图 CPU 延时:python scripts/run_ead_cpu.py
GPU 上训练(延时与训练质量无关,少步即可),搬到 CPU 测 score_large 整图卷积延时(目标<2s)。"""
import time
import glob
import torch
from aoi.efficientad import EfficientADDetector, OUT, _MEAN, _STD
from eval.mvtec import _load_img_native

SIZE = 2500


def to_cpu(det):
    det.teacher = det.teacher.cpu()
    det.student = det.student.cpu()
    det.ae = det.ae.cpu()
    det.t_mean = det.t_mean.cpu()
    det.t_std = det.t_std.cpu()
    det.q = tuple(q.cpu() for q in det.q)
    det._mean = _MEAN
    det._std = _STD
    det.device = "cpu"


def main():
    base = sorted(glob.glob("data/mvtec/*/train/good/*.png"))[0]
    img = _load_img_native(base)
    if max(img.shape[-2:]) < SIZE:
        import torch.nn.functional as F
        img = F.interpolate(img.unsqueeze(0), size=(SIZE, SIZE), mode="bilinear")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = EfficientADDetector(model_size="small", device=dev, train_steps=300)
    norms = [(img + torch.randn_like(img) * 0.02).clamp(0, 1) for _ in range(20)]
    det.fit_fewshot(norms, norms[:2])
    to_cpu(det)
    print(f"已搬到 CPU,测 2500² 整图卷积延时:", flush=True)
    print(f"{'max_size':10s} {'CPU延时':>10}", flush=True)
    for ms in [1024, 1280, 1536]:
        det.score_large(img, max_size=ms)              # 预热
        t0 = time.perf_counter()
        for _ in range(3):
            det.score_large(img, max_size=ms)
        lat = (time.perf_counter() - t0) / 3 * 1000
        tag = "✅<2s" if lat < 2000 else "❌超时"
        print(f"{ms:10d} {lat:8.0f}ms {tag}", flush=True)


if __name__ == "__main__":
    main()
