"""考场混合数据集(单产品目录,Web端可直接选):严格按赛题协议 100正常+30缺陷 一次迁移,
混合测试流出题。

数据来源(2026-08-30 换血):赛题原文"4．参考数据集"只点名两个公开集——**DAGM 2007**
和**MVTec AD**;此前用的是 Real-IAD 三类目(phone_battery/sim_card_set/pcb),其中
pcb 类目图片观感差、误报多,且三个都不在赛题参考数据集列表内。换成:
  - MVTec AD:hazelnut + cable(项目里验证最稳的两类,acc 0.92/0.93)
  - DAGM 2007:Class1(此前从未进过任何成绩单,补上赛题点名的这个数据集)
  - phone_best(leon 提供的手机屏数据集):油污/划痕/污渍

**如实说明**:phone_best 本身没有正常图(数据集结构性缺失)。改用 MSD 官方 good.zip
里的 20 张正常图(data/msd_good,此前两份衍生集都把这 20 张剥掉了,现补上)——仍不足
100 张,demo 展示用途够用,不代表评分协议"100 张正常图"的要求已被满足。

用法:PYTHONPATH=. python scripts/make_exam_data.py [输出目录=exam_data]
"""
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MV = Path("data/mvtec")
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
DAGM = Path("data/dagm/Class1")
MSD_GOOD = Path("data/msd_good/good")
PB = Path("data/phone_best")
RNG = random.Random(0)


def mvtec_pick(cat):
    """(正常图train池, [(缺陷图,掩膜)]池, 正常图test池) —— 三池均已固定种子打乱。"""
    root = MV / cat
    normals = sorted(root.glob("train/good/*.png"))
    ok_test = sorted(root.glob("test/good/*.png"))
    defs = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            for f in sorted(sub.glob("*.png")):
                m = GT / cat / "ground_truth" / sub.name / (f.stem + "_mask.png")
                if m.exists():
                    defs.append((f, m))
    for lst in (normals, defs, ok_test):
        RNG.shuffle(lst)
    return normals, defs, ok_test


def dagm_pick():
    """DAGM原生Train/Test各自都混着正常+缺陷,按有无Label拆开。"""
    def split(sub):
        normals, defs = [], []
        for f in sorted((DAGM / sub).glob("*.PNG")):
            m = DAGM / sub / "Label" / f"{f.stem}_label.PNG"
            (defs if m.exists() else normals).append((f, m if m.exists() else None))
        return normals, defs
    tr_norm, tr_def = split("Train")
    te_norm, te_def = split("Test")
    normals = tr_norm + te_norm
    defs = [(f, m) for f, m in tr_def + te_def]
    RNG.shuffle(normals); RNG.shuffle(defs)
    return normals, defs


def phone_best_defects(n):
    """(图路径, 矩形掩膜ndarray, 类型名) 列表。按文件名前缀均衡取三类。"""
    typ = {"Oil": "油污", "Scr": "划痕", "Sta": "污渍"}
    by = {k: [] for k in typ}
    for sp in ("train", "val", "test"):
        for f in sorted((PB / sp / "images").glob("*.jpg")):
            k = f.name[:3]
            if k in by and (PB / sp / "labels" / (f.stem + ".txt")).exists():
                by[k].append((f, PB / sp / "labels" / (f.stem + ".txt")))
    for k in by:
        RNG.shuffle(by[k])
    out, i = [], 0
    while len(out) < n and any(by.values()):
        k = list(typ)[i % 3]; i += 1
        if not by[k]:
            continue
        f, lab = by[k].pop()
        w, h = Image.open(f).size
        m = np.zeros((h, w), np.uint8)
        for line in open(lab):
            _, cx, cy, bw, bh = map(float, line.split())
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = 1
        out.append((f, m, typ[k]))
    return out


def main(out="exam_data"):
    root = Path(out) / "考场混合"
    for sub in ("normal", "defect", "mask", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    mv = {c: mvtec_pick(c) for c in ("hazelnut", "cable")}
    dg_norm, dg_def = dagm_pick()
    msd = sorted(MSD_GOOD.glob("*.png")); RNG.shuffle(msd)

    # 100 正常:hazelnut 30 + cable 30 + DAGM 25 + MSD官方正常图 15
    ni = 0
    for f in mv["hazelnut"][0][:30] + mv["cable"][0][:30]:
        shutil.copy(f, root / "normal" / f"n{ni:03d}.png"); ni += 1
    for f, _ in dg_norm[:25]:
        Image.open(f).convert("RGB").save(root / "normal" / f"n{ni:03d}.png"); ni += 1
    for f in msd[:15]:
        Image.open(f).convert("RGB").save(root / "normal" / f"n{ni:03d}.png"); ni += 1

    # 30 缺陷:hazelnut 8 + cable 8 + DAGM 7 + phone_best 7
    di = 0
    for cat, quota in (("hazelnut", 8), ("cable", 8)):
        for f, m in mv[cat][1][:quota]:
            shutil.copy(f, root / "defect" / f"d{di:03d}.png")
            mk = np.array(Image.open(m).convert("L")) > 0
            Image.fromarray((mk * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png"); di += 1
    for f, m in dg_def[:7]:
        Image.open(f).convert("RGB").save(root / "defect" / f"d{di:03d}.png")
        mk = np.array(Image.open(m).convert("L")) > 0
        Image.fromarray((mk * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png"); di += 1
    pb = phone_best_defects(7 + 45)
    for f, m, _ in pb[:7]:
        Image.open(f).save(root / "defect" / f"d{di:03d}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png"); di += 1

    # 测试流(赛题要求1000+):缺陷用完每类目fit之外的全部真实缺陷图(hazelnut/cable
    # 这两类MVTec原生缺陷总量本来就小,70/92张封顶,不足额靠DAGM/phone_best补不了假的,
    # 如实按真实存量出;正常图同理,test/good之外把train/good里fit没用到的部分也计入
    # (同一类目内部不与fit重复,不是造假),凑够量避开fit已用过的图
    rows, pool = [], []
    for cat in ("hazelnut", "cable"):
        for f, _ in mv[cat][1][8:]:
            pool.append((f, "缺陷"))
    for f, _ in dg_def[7:7 + 85]:
        pool.append((f, "缺陷"))
    for f, _, _ in pb[7:52]:
        pool.append((f, "缺陷"))
    for cat in ("hazelnut", "cable"):
        for f in mv[cat][2] + mv[cat][0][30:]:
            pool.append((f, "正常"))
    for f, _ in dg_norm[25:25 + 233]:
        pool.append((f, "正常"))
    for f in msd[15:20]:
        pool.append((f, "正常"))
    RNG.shuffle(pool)
    for i, (src, truth) in enumerate(pool):
        name = f"t{i:03d}.png"
        Image.open(src).convert("RGB").save(root / "test" / name)
        rows.append((name, truth))
    with open(root / "answer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "truth"]); w.writerows(rows)
    n_def = sum(1 for _, t in rows if t == "缺陷")
    print(f"完成:{root}  正常100 缺陷30(含phone_best 7) 测试{len(rows)}张(缺陷{n_def}/正常{len(rows)-n_def})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exam_data")
