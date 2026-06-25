"""EfficientAD OpenVINO CPU 延时:python scripts/run_ead_openvino_cpu.py
把 ST 主分支(teacher+student+归一+map_st)整体导出 OV,CPU 推理测 2500² 整图延时(目标<2s)。"""
import time
import glob
import torch
import torch.nn as nn
from aoi.efficientad import EfficientADDetector, OUT, _MEAN, _STD
from eval.mvtec import _load_img_native

SIZE = 2500


class STScore(nn.Module):
    """整图 ST 异常图:输入已归一化图 → map_st (B,1,h,w)。t_mean/t_std 烘焙为常量。"""
    def __init__(self, teacher, student, t_mean, t_std):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.register_buffer("tm", t_mean)
        self.register_buffer("ts", t_std)

    def forward(self, x):
        t = (self.teacher(x) - self.tm) / self.ts
        s = self.student(x)[:, :OUT]
        return ((t - s) ** 2).mean(1, keepdim=True)


def main():
    import openvino as ov
    base = sorted(glob.glob("data/mvtec/*/train/good/*.png"))[0]
    img = _load_img_native(base)
    import torch.nn.functional as F
    if max(img.shape[-2:]) < SIZE:
        img = F.interpolate(img.unsqueeze(0), size=(SIZE, SIZE), mode="bilinear")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = EfficientADDetector(model_size="small", device=dev, train_steps=300)
    norms = [(img + torch.randn_like(img) * 0.02).clamp(0, 1) for _ in range(20)]
    det.fit_fewshot(norms, norms[:2])

    mod = STScore(det.teacher.cpu().eval(), det.student.cpu().eval(),
                  det.t_mean.cpu(), det.t_std.cpu()).eval()
    print(f"{'max_size':10s} {'OV-CPU延时':>12}", flush=True)
    for ms in [1024, 1280, 1536]:
        x = img.unsqueeze(0)
        if max(x.shape[-2:]) > ms:
            x = F.interpolate(x, scale_factor=ms / max(x.shape[-2:]), mode="bilinear")
        xn = ((x - _MEAN) / _STD)
        ov_model = ov.convert_model(mod, example_input=xn)
        compiled = ov.Core().compile_model(ov_model, "CPU")
        compiled(xn.numpy())                              # 预热
        t0 = time.perf_counter()
        for _ in range(3):
            compiled(xn.numpy())
        lat = (time.perf_counter() - t0) / 3 * 1000
        tag = "✅<2s" if lat < 2000 else "❌超时"
        print(f"{ms:10d} {lat:10.0f}ms {tag}", flush=True)


if __name__ == "__main__":
    main()
