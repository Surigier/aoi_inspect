"""单域对照实验:验证§1.7"跨数据集混合导致阈值失配"的诊断——把MVTec/DAGM/手机屏
各自独立建一次100正常+30缺陷的迁移集+独立测试流,不再互相混域,看各自能否恢复到
接近历史真实水平。复用 make_exam_data.py 的 mvtec_pick/dagm_pick/phone_best_defects。

用法:PYTHONPATH=. python scripts/make_domain_exam.py mvtec|dagm|phone [输出目录=exam_data]
"""
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
import make_exam_data as m

RNG = random.Random(1)   # 独立种子,避免和考场混合那次共用状态


def dump(root, normals, defects_masks, test_pool):
    """normals: [path]; defects_masks: [(path,mask_or_None)]; test_pool: [(path,truth)]"""
    for sub in ("normal", "defect", "mask", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(normals):
        Image.open(f).convert("RGB").save(root / "normal" / f"n{i:03d}.png")
    di = 0
    for f, mk in defects_masks:
        Image.open(f).convert("RGB").save(root / "defect" / f"d{di:03d}.png")
        if mk is not None:
            m_ = np.array(Image.open(mk).convert("L")) > 0 if isinstance(mk, (str, Path)) else mk > 0
            Image.fromarray((m_ * 255).astype(np.uint8)).save(root / "mask" / f"d{di:03d}.png")
        di += 1
    rows = []
    for i, (f, truth) in enumerate(test_pool):
        name = f"t{i:03d}.png"
        Image.open(f).convert("RGB").save(root / "test" / name)
        rows.append([name, truth])
    with open(root / "answer.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "truth"]); w.writerows(rows)
    ndef = sum(1 for _, t in test_pool if t == "缺陷")
    print(f"{root}: 正常{len(normals)} 缺陷{len(defects_masks)} 测试{len(test_pool)}(缺陷{ndef}/正常{len(test_pool)-ndef})")


def build_mvtec(root):
    haz = m.mvtec_pick("hazelnut")
    cab = m.mvtec_pick("cable")
    normals = haz[0][:50] + cab[0][:50]
    defects = [(f, mk) for f, mk in haz[1][:15]] + [(f, mk) for f, mk in cab[1][:15]]
    pool = []
    for f, _ in haz[1][15:]: pool.append((f, "缺陷"))
    for f, _ in cab[1][15:]: pool.append((f, "缺陷"))
    for f in haz[2] + haz[0][50:]: pool.append((f, "正常"))
    for f in cab[2] + cab[0][50:]: pool.append((f, "正常"))
    RNG.shuffle(pool)
    dump(root, normals, defects, pool)


def build_dagm(root):
    norm, defs = m.dagm_pick()
    normals = [f for f, _ in norm[:100]]
    defects = defs[:30]
    pool = [(f, "缺陷") for f, _ in defs[30:]] + [(f, "正常") for f, _ in norm[100:100 + 400]]
    RNG.shuffle(pool)
    dump(root, normals, defects, pool)


def build_phone(root):
    msd = sorted(m.MSD_GOOD.glob("*.png")); RNG.shuffle(msd)
    normals = msd[:15]
    pb = m.phone_best_defects(30 + 150)
    defects = [(f, mk) for f, mk, _ in pb[:30]]
    pool = [(f, "缺陷") for f, _, _ in pb[30:180]] + [(f, "正常") for f in msd[15:20]]
    RNG.shuffle(pool)
    dump(root, normals, defects, pool)


if __name__ == "__main__":
    domain = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "exam_data") / f"单域_{domain}"
    {"mvtec": build_mvtec, "dagm": build_dagm, "phone": build_phone}[domain](out)
