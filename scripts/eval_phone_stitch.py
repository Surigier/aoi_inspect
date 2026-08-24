"""手机屏数据拼成 2500² 大图跑一遍——对齐赛题的真实输入形态。

出题方口径:2500×2500 的图由 1024² 量级的小图拼接而成。这里把4张手机屏图各放大到
1250² 摆成 2×2,拼成 2500×2500,GT框一并变换到大图坐标系。

**这一跑要回答三件事,别拿它讲精度故事**:
  ①2500²真尺寸下的延时(检测时间占竞赛得分30%,比准确率还重)
  ②拼接布局下定位还指不指得对位置(框命中@0.5)
  ③**四个象限的命中率均不均匀**——如果角落/接缝处系统性变差,说明分块或感受野有问题,
    这是拼接图独有的失效模式,单图测不出来。

**正常图严重饥饿的老问题在这里更严重**:全库只有7张正常图(还是增广重复),拼出来的
大图看着有20张,底下其实还是那2张原图的排列组合,多样性是假的。所以本脚本报的是
**下界+形态压测**,不是精度结论。图级准确率照样不可测(测试侧没有正常样本)。

用法:PYTHONPATH=. python scripts/eval_phone_stitch.py
"""
import glob
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import gt_boxes, box_hit

ROOT = "data/phone"
BIG, TILE = 2500, 1250                       # 2×2 的 1250² 块拼成 2500²


def _load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _tile_mask(shape_hw, lab_path):
    H, W = shape_hw
    m = np.zeros((H, W), np.uint8)
    if not os.path.exists(lab_path):
        return m
    for line in open(lab_path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = [float(x) for x in p[1:5]]
        x0, x1 = int((cx - w / 2) * W), int((cx + w / 2) * W)
        y0, y1 = int((cy - h / 2) * H), int((cy + h / 2) * H)
        m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 1
    return m


def _stitch(paths):
    """4条路径 → (3,2500,2500) 大图 + (2500,2500) GT掩膜。象限顺序:左上/右上/左下/右下。"""
    big = torch.zeros(3, BIG, BIG)
    gt = np.zeros((BIG, BIG), np.uint8)
    for k, f in enumerate(paths[:4]):
        img = _load(f)
        mk = _tile_mask(img.shape[-2:], f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        t = F.interpolate(img[None], size=(TILE, TILE), mode="bilinear", align_corners=False)[0]
        tm = F.interpolate(torch.from_numpy(mk.astype(np.float32))[None, None],
                           size=(TILE, TILE), mode="nearest")[0, 0].numpy() > 0.5
        r, c = k // 2, k % 2
        big[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        gt[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = tm.astype(np.uint8)
    return big, gt


def _defect_files(split, want, skip=0):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{split}/images/*"))[skip:]:
        lab = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lab) and os.path.getsize(lab) > 0:
            out.append(f)
        if len(out) >= want:
            break
    return out


def _normal_files():
    out = []
    for split in ("train", "valid", "test"):
        for f in sorted(glob.glob(f"{ROOT}/{split}/labels/*.txt")):
            if os.path.getsize(f) == 0:
                out.append(f.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg")
    return out


def _quadrant(box, ):
    """框中心落在哪个象限(0左上/1右上/2左下/3右下)——看拼接布局下命中率均不均匀。"""
    x0, y0, x1, y1 = box
    return (0 if (y0 + y1) / 2 < 0.5 else 2) + (0 if (x0 + x1) / 2 < 0.5 else 1)


def main():
    rng = random.Random(0)
    torch.manual_seed(0)
    nf = _normal_files()
    print(f"源正常图 {len(nf)} 张(排列组合出大图,多样性是假的——见模块说明)", flush=True)
    norm_big = [_stitch([nf[(i * 4 + k) % len(nf)] for k in range(4)])[0] for i in range(20)]

    fit_f = _defect_files("train", 32)
    fit_pairs = [_stitch(fit_f[i * 4:(i + 1) * 4]) for i in range(len(fit_f) // 4)]
    test_f = _defect_files("valid", 80)
    print(f"拼接后:正常大图 {len(norm_big)} / fit缺陷大图 {len(fit_pairs)} / 测试缺陷大图 {len(test_f)//4}",
          flush=True)

    t0 = time.time()
    det = CompetitionLargeDetector(compile_infer=True)
    det.fit_fewshot(norm_big, [i for i, _ in fit_pairs], defect_masks=[m for _, m in fit_pairs])
    print(f"fit完成 {time.time()-t0:.0f}s  type_head={'就绪' if det.type_head else '未启用'}", flush=True)
    del norm_big, fit_pairs

    warm, _ = _stitch(test_f[:4])
    for _ in range(3):
        det.locate(warm)                                   # 预热,不计时
    del warm

    hits, lat, n_det, n_img = [], [], 0, 0
    q_hit, q_tot = [0] * 4, [0] * 4
    types = {}
    for i in range(len(test_f) // 4):
        img, gt = _stitch(test_f[i * 4:(i + 1) * 4])
        t1 = time.time()
        o = det.locate(img)
        lat.append((time.time() - t1) * 1000)
        n_img += 1
        n_det += bool(o["is_defect"])
        if o["is_defect"]:
            types[o["defect_type"]] = types.get(o["defect_type"], 0) + 1
        if o.get("mask") is None:
            hits.append(0.0)
            continue
        mk = o["mask"]
        gt_r = (F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                              size=mk.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        gbs = gt_boxes(gt_r)
        h = box_hit(o["boxes"], gbs)
        hits.append(h if h is not None else 0.0)
        H, W = mk.shape
        for g in gbs:                                      # 逐框记象限,看拼接布局均不均匀
            q = _quadrant((g[0] / W, g[1] / H, g[2] / W, g[3] / H))
            q_tot[q] += 1
            q_hit[q] += any(box_hit([p], [g]) == 1.0 for p in o["boxes"])
        del img, gt

    print(f"\n=== 2500²拼接大图(4×1250²,正常图源仅{len(nf)}张 → 下界+形态压测) ===", flush=True)
    print(f"检出率      {n_det}/{n_img} = {n_det/max(n_img,1):.1%}", flush=True)
    print(f"框命中@0.5  {np.mean(hits):.3f}", flush=True)
    names = ["左上", "右上", "左下", "右下"]
    print("各象限框命中 " + "  ".join(
        f"{names[q]}:{q_hit[q]}/{q_tot[q]}={q_hit[q]/max(q_tot[q],1):.0%}" for q in range(4)), flush=True)
    print(f"延时(2500²真尺寸)  中位={np.median(lat):.0f}ms  p90={np.percentile(lat,90):.0f}ms  "
          f"最大={max(lat):.0f}ms   预算=200ms", flush=True)
    print(f"类型分布 {types}", flush=True)
    print("注:图级准确率不可测——测试侧无正常样本(7张源正常图全部用于fit)。", flush=True)
    print("STITCH OK", flush=True)


if __name__ == "__main__":
    main()
