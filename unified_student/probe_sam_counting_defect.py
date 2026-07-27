"""关键验证:逻辑异常缺陷图的实例数,相对正常图基线,偏离幅度有没有明显超过正常图
自己的分割噪声(否则这个信号会被SAM自身的分割不确定性淹没,做不出真实区分度)。
同时测一下everything模式的真实延时(box-prompted精修~19-38ms,everything模式是
完全不同的开销量级,之前没测过)。

只测昨天SAM稳定性检查里"较稳"的4类(breakfast_box/juice_bottle/pushpins/
splicing_connectors),screw_bag(CV=0.200不稳)不在这次验证范围内。

用法:PYTHONPATH=. python unified_student/probe_sam_counting_defect.py

【已验证,判负】延时是决定性问题:4类everything模式延时全部在5.9~7.2秒/张,是
190ms预算的30~36倍,没有例外——不管区分度如何,这个延时量级已经不可用。区分度
本身也不稳定(juice_bottle 9/10张缺陷明显偏离,但breakfast_box 0/10、pushpins
1/10、splicing_connectors 2/10)。**结论:MobileSAM everything模式做实例计数,
这个具体实现方案因延时不可行,主要卡在延时不是精度。"计数"这个思路本身没有被
证伪(juice_bottle证明信号有时候是真实存在的),但SAM everything模式这个实现
方式必须放弃,如果要继续走计数路线,得换更便宜的实现(比如crop_cascade.py已有的
ECC对齐模板残差+连通域计数,量级应该接近box-prompted SAM的~20-40ms,未测)。**
默认不接入competition.py,代码留opt-in研究件。
"""
import glob
import time
import numpy as np
from pathlib import Path
from ultralytics import SAM

WEIGHTS = str(Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt")
MIN_AREA_FRAC = 0.001
CATS = ["breakfast_box", "juice_bottle", "pushpins", "splicing_connectors"]
N_NORMAL = 10
N_DEFECT = 10


def count_instances(model, img_path):
    res = model(img_path, verbose=False)[0]
    if res.masks is None:
        return 0
    H, W = res.orig_shape
    areas = res.masks.data.sum(dim=(1, 2)).cpu().numpy()
    valid = areas >= (MIN_AREA_FRAC * H * W)
    return int(valid.sum())


def main():
    model = SAM(WEIGHTS)
    print("延时+区分度联合验证(缺陷图实例数 vs 正常基线,是否超出正常图自身噪声)", flush=True)
    for cat in CATS:
        normal_files = sorted(glob.glob(f"data/_dl/mvtec_loco/{cat}/train/good/*.png"))[:N_NORMAL]
        defect_files = sorted(glob.glob(f"data/_dl/mvtec_loco/{cat}/test/logical_anomalies/*.png"))[:N_DEFECT]
        if not normal_files or not defect_files:
            print(f"{cat}: 数据不足,跳过", flush=True)
            continue

        t0 = time.perf_counter()
        normal_counts = [count_instances(model, p) for p in normal_files]
        lat_per_img = (time.perf_counter() - t0) * 1000 / len(normal_files)
        mu, sd = np.mean(normal_counts), max(np.std(normal_counts), 1e-6)

        defect_counts = [count_instances(model, p) for p in defect_files]
        z = [(c - mu) / sd for c in defect_counts]
        n_sep = sum(1 for zz in z if abs(zz) >= 2.0)          # |z|>=2 才算"明显超出正常噪声"
        print(f"{cat:22s} 正常mean={mu:.1f}±{sd:.1f} | 缺陷实例数={defect_counts} "
              f"| |z|>=2的图={n_sep}/{len(defect_counts)} | everything模式单图延时≈{lat_per_img:.0f}ms",
              flush=True)


if __name__ == "__main__":
    main()
