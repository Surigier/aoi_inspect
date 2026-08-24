"""类型归属重做:位置匹配的专用判别特征(不复用检测分支的分数)。

前三次失败的共同点:**都在复用检测分支的分数**。那些分数是为"检测"设计的
(EAD是通用异常检测器,对任何缺陷都强响应),不携带类型信息。实测数据:
  类目          EAD    色彩   尺寸   结构   谁赢   对不对
  hazelnut(外观) 3.90  0.18  0.06  3.27   EAD    ✅
  pill(色彩)    6.72  4.70  ~0    0.68   EAD    ❌
  carpet(色彩)  3.33  6.47  ~0    8.96   结构   ❌   ←结构对色块的响应比色彩分支还强
  metal_nut(色彩)2.01 6.60  ~0    1.63   色彩   ✅
正确分支只在2/4类拿到最高分。已排除:①EAD降级兜底(hazelnut崩0/18)②掩膜内打分
(pill分布一模一样)③动态范围归一(50%→17%,尺寸分支范围≤0导致除爆)。

本方案:locate()已给出缺陷掩膜M。对每种类型算**专门的判别特征**,并且统一跟
**正常图在同一块掩膜位置M上的表现**比(位置匹配的零假设),z分因此天然可比:
  色彩变化 → 掩膜内色度(a*,b*)相对正常图同位置的偏移
  常见外观 → 掩膜内梯度幅值(高频纹理扰动)相对正常图同位置的偏移
  缺件/逻辑 → 掩膜内WRN深层特征相对正常图同位置的距离
  尺寸偏差 → 整图前景面积相对正常分布(本质是全局量,无法位置匹配,保留全局z)

用法:PYTHONPATH=. python scripts/diag_type_locmatch.py
"""
import collections
import numpy as np
import torch
import torch.nn.functional as F
from aoi.competition import CompetitionLargeDetector, _down
from aoi.branches.color_ad import rgb_to_lab
from aoi.fusion import znorm
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color

JOBS = [("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]), "外观缺陷"),
        ("pill", lambda: prep_mvtec("pill", ["color"]), "色彩变化"),
        ("carpet", lambda: prep_mvtec_color("carpet")[:4], "色彩变化"),
        ("metal_nut", lambda: prep_mvtec_color("metal_nut")[:4], "色彩变化")]
SZ = 320


def shallow_feats(img):
    """(3,H,W)[0,1] → 在SZ²上算的逐像素特征:色度(a*,b*)、亮度L、梯度幅值。"""
    x = _down(img, SZ).cpu()                              # (1,3,SZ,SZ)
    lab = rgb_to_lab(x)[0]                                # (3,SZ,SZ) = L,a,b
    gray = x[0].mean(0, keepdim=True)[None]               # (1,1,SZ,SZ)
    gx = F.conv2d(gray, torch.tensor([[[[-1., 0., 1.]]]]), padding=(0, 1))
    gy = F.conv2d(gray, torch.tensor([[[[-1.], [0.], [1.]]]]), padding=(1, 0))
    grad = (gx ** 2 + gy ** 2).sqrt()[0, 0]               # (SZ,SZ)
    return lab[1:3], grad                                  # 色度(2,SZ,SZ), 梯度(SZ,SZ)


def mask_at(mask, sz=SZ):
    m = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                      size=(sz, sz), mode="area")[0, 0] > 0.02
    return m if m.any() else torch.ones(sz, sz, dtype=torch.bool)


def pooled(img_feats, m):
    """掩膜内均值:色度→2维向量,梯度→标量。"""
    chroma, grad = img_feats
    return chroma[:, m].mean(1), float(grad[m].mean())


def deep_pooled(det, img, m):
    """掩膜内WRN深层特征均值(结构性)。"""
    f = det._wrn_feats(img)                                # (C,g,g)
    mm = F.interpolate(m[None, None].float(), size=f.shape[-2:], mode="area")[0, 0] > 0.02
    if not mm.any():
        mm = torch.ones_like(mm)
    return f[:, mm].mean(1).detach().cpu()


def main():
    torch.manual_seed(0)
    tot = collections.Counter()
    for cat, prep, truth in JOBS:
        normals, fit_i, fit_m, test_defs = prep()
        det = CompetitionLargeDetector(train_steps=100)
        det.fit_fewshot(normals[:40], fit_i, defect_masks=fit_m)
        ref = normals[:30]                                  # 位置匹配的零假设样本
        ref_shallow = [shallow_feats(x) for x in ref]
        names = ["外观缺陷", "色彩变化", "尺寸偏差", "缺件/逻辑"]

        c_new, n = collections.Counter(), 0
        for img, _ in test_defs[:20]:
            o = det.locate(img)
            if not o["is_defect"] or o.get("mask") is None:
                continue
            n += 1
            m = mask_at(o["mask"])
            q_chroma, q_grad = pooled(shallow_feats(img), m)
            q_deep = deep_pooled(det, img, m)
            # 同一块掩膜位置上,正常图的分布 = 零假设
            r_chroma = torch.stack([pooled(f, m)[0] for f in ref_shallow])       # (R,2)
            r_grad = np.array([pooled(f, m)[1] for f in ref_shallow])            # (R,)
            r_deep = torch.stack([deep_pooled(det, x, m) for x in ref])          # (R,C)
            # 三个特征必须用**同一套**统计口径,否则不可比。统一为:
            #   z = (测试图到正常均值的距离 - 正常图之间的典型距离) / 正常距离的散布
            # 关键是**要减掉零假设的典型距离**:正常图彼此之间本来就有非零距离,
            # 而深层特征是768维、高维空间范数高度集中(典型距离大、散布极小),
            # 不减均值直接算 距离/散布 会把结构特征系统性放大到通吃(实测hazelnut
            # 和carpet被结构分支全吃:0/18、0/12)。
            def _z_vec(q, R):
                mu = R.mean(0)
                dn = (R - mu).norm(dim=1)                  # 零假设:正常图到均值的距离分布
                return (float((q - mu).norm()) - float(dn.mean())) / (float(dn.std()) or 1.0)
            z_color = _z_vec(q_chroma, r_chroma)           # 色彩:掩膜内色度偏移
            z_struct = _z_vec(q_deep, r_deep)              # 缺件/逻辑:掩膜内深层特征偏移
            z_appear = (q_grad - r_grad.mean()) / (r_grad.std() or 1.0)   # 外观:梯度幅值偏移(标量,同口径)
            # 尺寸:全局前景面积(本质全局量,保留原分支的全局z)
            z_dim = znorm(det.branches[2].score(img), *det.stats[2])
            zs = [z_appear, z_color, z_dim, z_struct]
            c_new[names[int(np.argmax(zs))]] += 1
        ok = c_new.get(truth, 0)
        tot["n"] += n; tot["ok"] += ok
        print(f"{cat}(真实={truth},检出{n}): 位置匹配特征={ok}/{n} {dict(c_new)}", flush=True)
    print(f"\n合计: 位置匹配特征 {tot['ok']}/{tot['n']}={tot['ok']/max(tot['n'],1):.0%}  (现状基线=50%)", flush=True)
    print("LOCMATCH OK", flush=True)


if __name__ == "__main__":
    main()
