"""考场混合test/里每张图来自哪个源数据集——只为按类目拆分实验结果看用,不重新生成图片。

复用 make_exam_data.py 的 mvtec_pick/dagm_pick/phone_best_defects 三个取数函数,原样重放
main() 里组装fit/test池的那部分逻辑(同一个RNG(0)实例、同样的调用顺序),这样第i次
进pool的图和当初生成exam_data时第i张test图是同一张,只是这次不落盘、只记类目。

用法:PYTHONPATH=. python scripts/tag_exam_category.py > exam_data/考场混合/category.csv
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import make_exam_data as m

m.RNG = random.Random(0)   # 重置成main()开始时的状态,和原始生成流程对齐

mv = {c: m.mvtec_pick(c) for c in ("hazelnut", "cable")}
dg_norm, dg_def = m.dagm_pick()
msd = sorted(m.MSD_GOOD.glob("*.png")); m.RNG.shuffle(msd)

ni = 0
for f in mv["hazelnut"][0][:30] + mv["cable"][0][:30]:
    ni += 1
for f, _ in dg_norm[:25]:
    ni += 1
for f in msd[:15]:
    ni += 1

di = 0
for cat, quota in (("hazelnut", 8), ("cable", 8)):
    for f, mk in mv[cat][1][:quota]:
        di += 1
for f, mk in dg_def[:7]:
    di += 1
pb = m.phone_best_defects(7 + 45)
for f, mk, _ in pb[:7]:
    di += 1

pool = []
for cat in ("hazelnut", "cable"):
    for f, _ in mv[cat][1][8:]:
        pool.append((cat, "缺陷"))
for f, _ in dg_def[7:7 + 85]:
    pool.append(("dagm", "缺陷"))
for f, _, _ in pb[7:52]:
    pool.append(("phone_best", "缺陷"))
for cat in ("hazelnut", "cable"):
    for f in mv[cat][2] + mv[cat][0][30:]:
        pool.append((cat, "正常"))
for f, _ in dg_norm[25:25 + 233]:
    pool.append(("dagm", "正常"))
for f in msd[15:20]:
    pool.append(("msd", "正常"))
m.RNG.shuffle(pool)

w = csv.writer(sys.stdout)
w.writerow(["file", "category", "truth"])
for i, (cat, truth) in enumerate(pool):
    w.writerow([f"t{i:03d}.png", cat, truth])
