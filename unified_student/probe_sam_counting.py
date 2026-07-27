"""计数/实例级路线的地基验证:MobileSAM(生产已在用,models/mobile_sam.pt)的
"everything"自动分割模式,能不能在LOCO"数量/排布"类目(pushpins/screw_bag等小物件
密集场景)上把实例分得稳——这是建计数机制前必须先确认的前提,不是新增依赖。

判据:同一类目的多张**正常图**(理论上物件数量应该完全一致),SAM分出来的"有效
实例数"(过滤掉背景/过小碎块后)变异系数(CV=std/mean)要低,才说明分割足够稳定,
可以拿来做计数比对;CV高说明SAM在这类场景下分割本身不稳,计数机制这条路此时
建立在不稳的地基上,不值得往下投入。

用法:PYTHONPATH=. python unified_student/probe_sam_counting.py

【已验证】5类目里4类稳定(breakfast_box CV=0.122/juice_bottle 0.114/pushpins
0.076/splicing_connectors 0.080),只有screw_bag不稳(CV=0.200,未查具体原因,
可能是螺丝粘连/反光导致过分或欠分)。地基本身没问题,但见probe_sam_counting_defect.py
——真正卡住这条路的是延时,不是分割稳定性。
"""
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import SAM

WEIGHTS = str(Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt")
MIN_AREA_FRAC = 0.001                      # 过滤掉小于图面积0.1%的碎块(噪声/纹理误分)
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
N_IMAGES = 10


def count_instances(model, img_path):
    res = model(img_path, verbose=False)[0]           # 无bbox/points → everything模式
    if res.masks is None:
        return 0
    H, W = res.orig_shape
    areas = res.masks.data.sum(dim=(1, 2)).cpu().numpy()  # 每个mask的像素数
    valid = areas >= (MIN_AREA_FRAC * H * W)
    return int(valid.sum())


def main():
    model = SAM(WEIGHTS)
    print(f"MobileSAM everything模式实例计数稳定性诊断(正常图,理论物件数应恒定)", flush=True)
    for cat in CATS:
        files = sorted(glob.glob(f"data/_dl/mvtec_loco/{cat}/train/good/*.png"))[:N_IMAGES]
        if not files:
            print(f"{cat}: 无数据,跳过", flush=True)
            continue
        counts = [count_instances(model, p) for p in files]
        cv = float(np.std(counts) / max(np.mean(counts), 1e-6))
        print(f"{cat:22s} 实例数={counts}  mean={np.mean(counts):.1f} std={np.std(counts):.2f} "
              f"CV={cv:.3f}  {'✅较稳' if cv < 0.15 else '❌不稳'}", flush=True)


if __name__ == "__main__":
    main()
