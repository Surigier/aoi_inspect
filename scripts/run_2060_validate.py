"""2060 租卡一键验证:真机延时(<200ms)+ 显存(<6GB)达标验证。生产配置(compile+双学生集成)。

无需数据集:延时与图像内容无关,脚本自造合成图(fit 质量无所谓,推理成本相同);
量最坏情况(强制全判缺陷→SAM 必触发)+ 含单图加载解码的端到端(赛题计时口径)。

租的机器上跑:
  cd aoi_inspect
  pip install -r requirements.txt        # torch 需带 CUDA(AutoDL 镜像一般自带)
  export HF_ENDPOINT=https://hf-mirror.com   # 国内机器下 timm 权重用镜像
  PYTHONPATH=. python scripts/run_2060_validate.py                # 生产配置
  PYTHONPATH=. python scripts/run_2060_validate.py --students 1   # 对照:单学生回退档
"""
import argparse
import time
import tempfile
from pathlib import Path
import numpy as np
import torch


def synth_img(h, w, seed):
    """合成'像产品图'的内容(渐变+矩形+轻噪声):压缩率接近真图,解码计时才真实。"""
    rng = np.random.RandomState(seed)
    gx = np.linspace(60, 180, w, dtype=np.float32)[None, :]
    gy = np.linspace(0, 40, h, dtype=np.float32)[:, None]
    base = (gx + gy)
    img = np.stack([base] * 3, -1)
    for _ in range(12):                                       # 一些"零件"矩形
        y0, x0 = rng.randint(0, h - h // 8), rng.randint(0, w - w // 8)
        hh, ww = rng.randint(h // 20, h // 8), rng.randint(w // 20, w // 8)
        img[y0:y0 + hh, x0:x0 + ww] = rng.randint(40, 220, 3)
    img += rng.randn(h, w, 3) * 4
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=2, help="EAD学生数(生产=2;超线可回退1)")
    ap.add_argument("--steps", type=int, default=1500, help="EAD训练步数(延时与此无关,快跑即可)")
    ap.add_argument("--n-timed", type=int, default=20)
    args = ap.parse_args()

    import cv2
    from PIL import Image
    from aoi.competition import CompetitionLargeDetector
    from aoi.imageio import load_fast

    dev_ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if dev_ok else "CPU"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if dev_ok else 0
    print(f"=== 2060 一键验证 ===")
    print(f"GPU: {name} | 显存 {total_gb:.1f}GB | torch {torch.__version__} | CUDA可用={dev_ok}", flush=True)
    if not dev_ok:
        print("!! 无CUDA,退出"); return

    # ── 合成 fit 集(内容无所谓,尺寸走小图省时;推理成本只取决于测试图) ──
    rng = np.random.RandomState(0)
    def t3(a):
        return torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32) / 255.0)
    normals = [t3(synth_img(640, 640, i)) for i in range(100)]
    defects = [t3(synth_img(640, 640, 1000 + i)) for i in range(30)]
    masks = []
    for i in range(30):                                       # 随机块状掩膜(训分割头用)
        m = np.zeros((256, 256), np.uint8)
        y, x = rng.randint(30, 200, 2); m[y:y + 40, x:x + 40] = 1
        masks.append(m)

    print(f"fit 中(students={args.students}, steps={args.steps},不计时)...", flush=True)
    t0 = time.perf_counter()
    det = CompetitionLargeDetector(train_steps=args.steps, compile_infer=True,
                                   ead_students=args.students)
    det.fit_fewshot(normals, defects, defect_masks=masks)
    print(f"fit 完成,耗时 {time.perf_counter()-t0:.0f}s(赛题不计时)", flush=True)

    # 强制全判缺陷 → segment+SAM+框全链必走 = 最坏情况延时(诚实上界)
    det.threshold = -1e9
    if getattr(det, "_dino", None) is not None:
        det._dino_thr = -1e9

    # ── 测试场景:方形2500²(最重)/ 手机PCB细长 / 窄条;PNG(慢解码)+JPG 双格式 ──
    tmp = Path(tempfile.mkdtemp(prefix="v2060_"))
    scenarios = [("方形2500x2500", 2500, 2500), ("细长3034x1586", 1586, 3034), ("窄条1600x720", 720, 1600)]
    torch.cuda.reset_peak_memory_stats()
    results = []
    for tag, h, w in scenarios:
        arr = synth_img(h, w, 7)
        for fmt in ("png", "jpg"):
            p = tmp / f"{tag}.{fmt}"
            Image.fromarray(arr).save(str(p), quality=92) if fmt == "jpg" else Image.fromarray(arr).save(str(p))
            for _ in range(5):                                # 预热(compile/cudnn autotune)
                det.locate(load_fast(str(p)))
            torch.cuda.synchronize()
            lats = []
            for _ in range(args.n_timed):
                t0 = time.perf_counter()
                img = load_fast(str(p))                       # 含加载+解码(赛题口径)
                det.locate(img)
                torch.cuda.synchronize()
                lats.append((time.perf_counter() - t0) * 1000)
            lats = np.array(lats)
            ok = lats.mean() < 200
            results.append(ok)
            print(f"{tag} [{fmt.upper()}] 端到端: 均值={lats.mean():.0f}ms 中位={np.median(lats):.0f} "
                  f"p90={np.percentile(lats,90):.0f} min={lats.min():.0f}  → {'✅<200ms' if ok else '❌超线'}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    vram_ok = peak < 5.5
    print(f"\n推理显存峰值: {peak:.2f}GB / {total_gb:.1f}GB  → {'✅' if vram_ok else '❌超6GB预算'}", flush=True)
    print(f"\n=== 终判: 延时 {sum(results)}/{len(results)} 场景达标 | 显存{'达标' if vram_ok else '超'} | "
          f"students={args.students} ===", flush=True)
    if not all(results):
        print("超线时依次尝试: --students 1 / 确认compile生效(看fit后无回退警告) / 调低max_size", flush=True)


if __name__ == "__main__":
    main()
