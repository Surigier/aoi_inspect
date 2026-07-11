"""集成生产化冒烟+延时门:①n_students=2 fit/score 正确性 ②真2500²大图 locate 延时(关键门)。
用法:PYTHONPATH=. python scripts/verify_ens_prod.py
"""
import glob
import time
import numpy as np
import torch
from aoi.efficientad import EfficientADDetector
from aoi.imageio import load_fast
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    normals = [_load_img(p, 640) for p in sorted(glob.glob("data/mvtec/hazelnut/train/good/*.png"))[:30]]
    # ① 正确性:2学生 fit + 双路径打分
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=1200, n_students=2)
    det.fit_fewshot(normals, None)
    assert det.pairs is not None and len(det.pairs) == 2, "pairs != 2"
    s1 = det._image_score(normals[0])[0]
    sl = det.score_large(normals[0])
    print(f"① 正确性 OK:pairs=2, _image_score={s1:.3f}, score_large={sl:.3f}", flush=True)
    # ② 延时门:真实近2500²大图(PKU-PCB),score_large 单学生 vs 双学生(共享教师)
    pcb = sorted(glob.glob("data/_dl/pku_pcb/**/*.jpg", recursive=True))[:3]
    if not pcb:
        print("② 无PKU-PCB图,跳过大图延时", flush=True)
        return
    img = load_fast(pcb[0])
    print(f"   大图尺寸: {tuple(img.shape)}", flush=True)
    for tag, pairs in [("单学生", det.pairs[:1]), ("双学生共享教师", det.pairs)]:
        det.pairs = pairs
        for _ in range(3):
            det.score_large(img)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            det.score_large(img)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"② {tag}: score_large={ms:.0f}ms  (2060估≈{ms*1.5:.0f}ms)", flush=True)


if __name__ == "__main__":
    main()
