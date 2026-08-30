"""考场混合数据集(单产品目录,Web端可直接选):严格按赛题协议 100正常+30缺陷 一次迁移,
混合测试流出题。混入四个手机部件域:Real-IAD phone_battery/sim_card_set/pcb(真实像素掩膜)
+ 手机屏 phone_best(油污/划痕/污渍,YOLO框转矩形掩膜)。

**如实说明**:phone_best 没有正常图(数据集结构性缺失,见交付文档),所以正常基准全部
来自三个 Real-IAD 类目;phone_best 只出缺陷图(fit 7张 + test 15张)。固定种子,可复现。

用法:PYTHONPATH=. python scripts/make_exam_data.py [输出目录=exam_data]
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RI = Path("data/_dl/Real-IAD")
RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
PB = Path("data/phone_best")
RNG = random.Random(0)


def realiad_pick(cat):
    d = json.load(open(RJ / f"{cat}.json"))
    ok_train = [x for x in d["train"] if x["anomaly_class"] == "OK"]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
    ok_test = [x for x in d["test"] if x["anomaly_class"] == "OK"]
    for lst in (ok_train, ng, ok_test):
        RNG.shuffle(lst)
    return ok_train, ng, ok_test


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
    cats = ["phone_battery", "sim_card_set", "pcb"]
    picked = {c: realiad_pick(c) for c in cats}

    # 100 正常:34/33/33
    ni = 0
    for c, quota in zip(cats, (34, 33, 33)):
        for x in picked[c][0][:quota]:
            shutil.copy(RI / c / x["image_path"], root / "normal" / f"n{ni:03d}.png"); ni += 1

    # 30 缺陷:Real-IAD 8/8/7 + phone_best 7
    di = 0
    for c, quota in zip(cats, (8, 8, 7)):
        for x in picked[c][1][:quota]:
            shutil.copy(RI / c / x["image_path"], root / "defect" / f"d{di:03d}.png")
            m = np.array(Image.open(RI / c / x["mask_path"]).convert("L")) > 0
            Image.fromarray((m * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png"); di += 1
    pb = phone_best_defects(7 + 15)
    for f, m, _ in pb[:7]:
        Image.open(f).save(root / "defect" / f"d{di:03d}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png"); di += 1

    # 测试流 300:缺陷 90(25/25/25 Real-IAD + 15 phone_best),正常 210(70/70/70)
    rows, pool = [], []
    for c, quota in zip(cats, (25, 25, 25)):
        for x in picked[c][1][30:30 + quota]:          # 避开fit缺陷
            pool.append((RI / c / x["image_path"], "缺陷"))
    for f, _, _ in pb[7:22]:
        pool.append((f, "缺陷"))
    for c in cats:
        for x in picked[c][2][:70]:
            pool.append((RI / c / x["image_path"], "正常"))
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
