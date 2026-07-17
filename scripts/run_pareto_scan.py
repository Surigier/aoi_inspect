"""真实Pareto扫描:students(1/2)×DINO(on/off)×SAM(on/off)×max_pixels(700k/900k/1100k/1400k)
= 32组合。精度用AD2真实掩膜数据测(fit一次,事后切换config重新推理,不重复训练);
延时用真实原生文件测(3种真实形状×2种格式,禁止合成/缩放重建)。

架构前提(已验证):max_pixels只影响EAD检测分支面积,不影响WRN分割头(seg_in定尺寸)。
故:
  - SAM×max_pixels(8组合)决定纯定位IoU/框命中(SAM是唯一直接影响像素定位的开关)
  - students×DINO×max_pixels(16组合)决定图级acc/召回(检测门控)
  - 32组合的"含漏检IoU" = 纯定位IoU(取决于SAM) 在 该组合判为缺陷 的图上取值,否则0

真实延时探针(3形状,来自磁盘真实文件,不缩放重建):
  方形2500²  = PKU-PCB(2282×2248, JPEG原生; 另存PNG测格式轴)
  中框宽图    = AD2 sheet_metal(4224×1056, PNG原生; 另存JPEG测格式轴)
  手机长条    = AD2 vial(1400×1900, PNG原生; 另存JPEG测格式轴)

用法:PYTHONPATH=. python scripts/run_pareto_scan.py
"""
import glob
import random
import time
import itertools
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import gt_boxes, box_hit, _read

AD2 = Path("data/mvtec_ad_2")
PKU = Path("data/_dl/pku_pcb")
HW = (256, 256)
STUDENTS_OPTS = (1, 2)
DINO_OPTS = (True, False)
SAM_OPTS = (True, False)
MP_OPTS = (700_000, 900_000, 1_100_000, 1_400_000)
LAT_BUDGET_HARD = 190.0


def prep_ad2(cat, n_norm=100, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    m = lambda p: _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test_defs = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + 40]]
    goods = [(load_fast(p), None) for p in sorted(glob.glob(str(root / "test_public/good/*.png")))[:40]]
    return normals, fit_i, fit_m, test_defs, goods


def iou(pred, gt):
    p = pred.astype(bool)
    TP = int((p & (gt == 1)).sum()); FP = int((p & (gt == 0)).sum()); FN = int((~p & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def _toggle(det, students, dino_on, sam_on, max_pixels):
    """post-hoc切换配置(不重新fit)。返回复原函数。"""
    ead = det.branches[0].det.det
    tiled = det.branches[0].det
    orig = {
        "pairs": ead.pairs, "dino": det._dino, "sam": det.sam,
        "max_pixels": tiled.max_pixels,
    }
    if ead.pairs:
        ead.pairs = ead.pairs[:students]
        ead.student, ead.ae, ead.q = ead.pairs[0]
    if not dino_on:
        det._dino = None
    if not sam_on:
        det.sam = None
    tiled.max_pixels = max_pixels

    def restore():
        ead.pairs = orig["pairs"]
        if orig["pairs"]:
            ead.student, ead.ae, ead.q = orig["pairs"][0]
        det._dino = orig["dino"]
        det.sam = orig["sam"]
        tiled.max_pixels = orig["max_pixels"]
    return restore


def eval_accuracy(det, test_defs, test_goods):
    """跑一遍locate(),返回(纯定位IoU均值, 框命中均值, 图级acc, 含漏检IoU均值)。"""
    ious_pure, hits, gated = [], [], []
    n_ok = 0; total = 0
    for img, gt in test_defs:
        o = det.locate(img)
        total += 1
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        v = TP / max(TP + FP + FN, 1)
        ious_pure.append(v)
        if o["is_defect"]:
            n_ok += 1; gated.append(v)
            h = box_hit(o["boxes"], gt_boxes(gt))
            if h is not None:
                hits.append(h)
        else:
            gated.append(0.0); hits.append(0.0)
    for img, _ in test_goods:
        total += 1
        if not det.locate(img)["is_defect"]:
            n_ok += 1
    return float(np.mean(ious_pure)), float(np.mean(hits)), n_ok / max(total, 1), float(np.mean(gated))


def measure_latency(det, probe_files, n_timed=8):
    """真实文件端到端p90延时(load_fast解码+locate,强制最坏链全判缺陷→SAM/框必走)。"""
    thr, dthr = det.threshold, getattr(det, "_dino_thr", None)
    det.threshold = -1e9
    if getattr(det, "_dino", None) is not None:
        det._dino_thr = -1e9
    lats = []
    for pf in probe_files:
        for _ in range(2):
            det.locate(load_fast(str(pf)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_timed):
            det.locate(load_fast(str(pf)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / n_timed * 1000
        lats.append(per)
    det.threshold = thr
    if dthr is not None:
        det._dino_thr = dthr
    return float(np.mean(lats)), float(np.percentile(lats, 90)) if len(lats) > 1 else lats[0]


def prep_probe_files(tmpdir):
    """真实原生文件,3形状×2格式=6个探针,禁止缩放重建。"""
    tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    files = {}
    # 方形2500²:PKU-PCB真实照片(原生JPEG),另存PNG测格式轴
    pku = sorted(glob.glob(str(PKU / "images" / "*.jpg")))
    square_src = max(pku[:50], key=lambda p: Path(p).stat().st_size)  # 挑一张真实较大的
    im = Image.open(square_src).convert("RGB")
    files["square_jpg"] = square_src
    p = tmpdir / "square.png"; im.save(str(p)); files["square_png"] = str(p)
    # 中框宽图:AD2 sheet_metal(原生PNG 4224x1056),另存JPEG
    wide_src = sorted(glob.glob(str(AD2 / "sheet_metal/train/good/*.png")))[0]
    files["wide_png"] = wide_src
    im = Image.open(wide_src).convert("RGB")
    p = tmpdir / "wide.jpg"; im.save(str(p), quality=92); files["wide_jpg"] = str(p)
    # 手机长条:AD2 vial(原生PNG 1400x1900),另存JPEG
    strip_src = sorted(glob.glob(str(AD2 / "vial/train/good/*.png")))[0]
    files["strip_png"] = strip_src
    im = Image.open(strip_src).convert("RGB")
    p = tmpdir / "strip.jpg"; im.save(str(p), quality=92); files["strip_jpg"] = str(p)
    for k, v in files.items():
        with Image.open(v) as _im:
            print(f"  探针[{k:10s}] {v}  尺寸={_im.size}", flush=True)
    return files


def main():
    torch.manual_seed(0)
    import tempfile
    probe_files = prep_probe_files(tempfile.mkdtemp(prefix="pareto_probe_"))

    print("\n=== 精度轴:AD2真实掩膜数据(fit一次,post-hoc切换SAM×max_pixels测纯定位)===", flush=True)
    acc_cats = ["sheet_metal", "walnuts", "fruit_jelly"]
    acc_results = {}   # (sam, mp) -> {cat: (pure_iou, box_hit)}
    det_cache = {}     # cat -> (det, test_defs, test_goods)
    for cat in acc_cats:
        normals, fit_i, fit_m, test_defs, test_goods = prep_ad2(cat)
        det = CompetitionLargeDetector(ead_students=2)   # 全功能fit一次
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        det_cache[cat] = (det, test_defs, test_goods)
        for sam_on in SAM_OPTS:
            for mp in MP_OPTS:
                restore = _toggle(det, students=2, dino_on=True, sam_on=sam_on, max_pixels=mp)
                pure_iou, bh, img_acc, gated = eval_accuracy(det, test_defs, test_goods)
                restore()
                acc_results.setdefault((sam_on, mp), {})[cat] = (pure_iou, bh)
                print(f"  {cat:14s} SAM={sam_on!s:5s} mp={mp//1000}k  纯定位IoU={pure_iou:.3f} 框={bh:.3f}", flush=True)

    print("\n=== 检测轴:students×DINO×max_pixels 对图级acc/含漏检的影响(复用同一批det)===", flush=True)
    det_acc = {}  # (students, dino, mp) -> {cat: (img_acc, gated_iou)}
    for cat in acc_cats:
        det, test_defs, test_goods = det_cache[cat]
        for students in STUDENTS_OPTS:
            for dino_on in DINO_OPTS:
                for mp in MP_OPTS:
                    restore = _toggle(det, students=students, dino_on=dino_on, sam_on=True, max_pixels=mp)
                    _, _, img_acc, gated = eval_accuracy(det, test_defs, test_goods)
                    restore()
                    det_acc.setdefault((students, dino_on, mp), {})[cat] = (img_acc, gated)

    print("\n=== 延时轴:32组合 × 6个真实探针文件(p90端到端)===", flush=True)
    # 用sheet_metal的det做延时基准(架构一致,延时不依赖具体类别内容)
    lat_det, _, _ = det_cache["sheet_metal"]
    lat_results = {}
    for students, dino_on, sam_on, mp in itertools.product(STUDENTS_OPTS, DINO_OPTS, SAM_OPTS, MP_OPTS):
        restore = _toggle(lat_det, students, dino_on, sam_on, mp)
        mean_ms, p90_ms = measure_latency(lat_det, list(probe_files.values()))
        restore()
        lat_results[(students, dino_on, sam_on, mp)] = (mean_ms, p90_ms)
        print(f"  s={students} dino={dino_on!s:5s} sam={sam_on!s:5s} mp={mp//1000:4d}k  "
              f"均值={mean_ms:.0f}ms p90={p90_ms:.0f}ms", flush=True)

    print("\n=== Pareto前沿:p90<190ms 中,含漏检IoU(3类均值)最高的组合 ===", flush=True)
    table = []
    for students, dino_on, sam_on, mp in itertools.product(STUDENTS_OPTS, DINO_OPTS, SAM_OPTS, MP_OPTS):
        mean_ms, p90_ms = lat_results[(students, dino_on, sam_on, mp)]
        pure_ious = [acc_results[(sam_on, mp)][c][0] for c in acc_cats]
        boxes = [acc_results[(sam_on, mp)][c][1] for c in acc_cats]
        img_accs = [det_acc[(students, dino_on, mp)][c][0] for c in acc_cats]
        gateds = [det_acc[(students, dino_on, mp)][c][1] for c in acc_cats]
        # 用SAM×mp的纯定位IoU替换gated中的定位部分(det_acc测的是sam=True下的gated,这里做近似融合)
        table.append({
            "students": students, "dino": dino_on, "sam": sam_on, "mp": mp,
            "p90_ms": p90_ms, "pure_iou": float(np.mean(pure_ious)),
            "box_hit": float(np.mean(boxes)), "img_acc": float(np.mean(img_accs)),
            "gated_iou_approx": float(np.mean(gateds)) if sam_on else float(np.mean(pure_ious)) * float(np.mean(img_accs)),
        })
    ok = [r for r in table if r["p90_ms"] < LAT_BUDGET_HARD]
    ok.sort(key=lambda r: -r["gated_iou_approx"])
    for r in (ok or table)[:8]:
        print(f"  students={r['students']} dino={r['dino']!s:5s} sam={r['sam']!s:5s} mp={r['mp']//1000}k  "
              f"p90={r['p90_ms']:.0f}ms  纯定位IoU={r['pure_iou']:.3f} 框={r['box_hit']:.3f} "
              f"图级acc={r['img_acc']:.3f} 含漏检≈{r['gated_iou_approx']:.3f}", flush=True)
    if not ok:
        print("  !! 无组合满足p90<190ms,报告全部组合按延时升序供参考", flush=True)


if __name__ == "__main__":
    main()
