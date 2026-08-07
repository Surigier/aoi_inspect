"""验证INP-Former(CVPR2025,MIT协议,github.com/luow23/INP-Former)能不能给我们的
图级检测提供一个有用的新信号。机制和EAD/DINO都不同:不比较test图和外部参考图,而是
每张图自己内部提取"内在正常原型"(INP)做重建,重建误差=异常分数——单张图一次前向
推理,不随k-shot数量增加计算量(这点上比VisionAD那种O(N²)最近邻检索便宜得多)。

用官方Few-Shot(k=4) MVTec-AD checkpoint(hazelnut/pill/cable/carpet/leather/
metal_nut/wood都在其训练用的标准15类item_list里,可以直接测,不用自己训练),接入
max(z_EAD, z_DINO, z_INP)融合模式——和当年给cable加DINO门、以及本session其它探针
(probe_residual_gate.py等)同一套融合模式。

预处理严格复现官方get_data_transforms:Resize(448,448)→CenterCrop(392)→ImageNet
Normalize。INP-Former的异常图原生就是392×392(crop尺寸),对回原图坐标时,把392的图
放回448画布对应的中心裁剪位置,其余(约12.5%的边框)补0,再插值到原图分辨率——这意味着
画面最外圈一圈INP-Former完全看不到,这是已知的局限,不是bug。

成功判据:同session口径,gated IoU的median(Δ)>=0.005且min(Δ)>=-0.01。

用法:PYTHONPATH=. python inp_former_probe/probe_inp.py

【已验证,判负,2026-07-29】7类(hazelnut/cable/pill/carpet/leather/metal_nut/wood)
ΔIoU=[0.000,-0.111,0.000,0.000,-0.130,0.000,-0.143],median=0.000 mean=-0.055
min=-0.143,四项判据全部不过关。**不是"没起作用"的中性结果,是真实倒退**:cable/
leather/wood三类接入INP-Former信号后,把原本正确的EAD+DINO融合判断直接压过去,
分数实打实变差——是本session测过的所有机制里最差的一次(其余8条路线至少中位数
接近0)。

初步归因(未深究,记录留痕):INP-Former的"内在正常原型"机制是从测试图自身patch
分布里找主导模式当"正常"基准,前提假设是"图里大部分区域本来就是正常的,异常只是
局部小块"——这个假设和MVTec-AD原论文自己的评测设置(每类单独训练/评测)一致,但在
我们的融合场景下,它给出的分数尺度、置信区间和EAD/DINO完全不是一回事,直接max()
过去等于让一个没有专门针对我们数据分布校准过的异质信号源，越权推翻两个已经经过
本项目大量调优的信号源的判断——这类"外来信号不经消化直接max()进决策"的做法,本
session另外几条判负路线(残差信号、蒸馏探针)也不同程度地暴露了类似问题。默认不
接入competition.py,代码留opt-in研究件。
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INP_ROOT = Path("/tmp/claude-1000/-home-srj-yolo/d8e53301-3b28-4836-8a41-ee22b204462e/scratchpad/INP-Former")
sys.path.insert(0, str(INP_ROOT))
from models import vit_encoder  # noqa: E402
from models.uad import INP_Former  # noqa: E402
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block  # noqa: E402


def cal_anomaly_maps(fs_list, ft_list, out_size):
    """照抄utils.py:153的原版实现(MIT协议,原样复用),绕开该文件顶部adeval这个
    pypi mirror上403拉不到的依赖(我们完全用不到它的eval工具函数)。"""
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    a_map_list = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map_list.append(a_map)
    return torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True), a_map_list

from aoi.competition import CompetitionLargeDetector
from aoi.fewshot import FewShotAdapter
from aoi.seg_head import map_to_boxes, merge_boxes
from global_context.eval_global_branch import prep_mvtec, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = INP_ROOT / (
    "saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14"
    "_Resize=448_Crop=392_INP_num=6/model.pth"
)
RESIZE, CROP = 448, 392
PAD = (RESIZE - CROP) // 2
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def build_inp_model():
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    inp = nn.ParameterList([nn.Parameter(torch.randn(6, embed_dim))])
    inp_extractor = nn.ModuleList([Aggregation_Block(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-8))])
    inp_decoder = nn.ModuleList([Prototype_Block(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(8)])
    model = INP_Former(encoder=encoder, bottleneck=bottleneck, aggregation=inp_extractor,
                        decoder=inp_decoder, target_layers=target_layers, remove_class_token=True,
                        fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder,
                        prototype_token=inp)
    model.load_state_dict(torch.load(CKPT, map_location=DEV), strict=True)
    model.to(DEV).eval()
    return model


@torch.no_grad()
def inp_anomaly_map(model, img):
    """img: (3,H,W) float[0,1] cpu tensor -> (H,W) numpy异常图,已放回原图坐标,边框补0。"""
    H, W = img.shape[-2], img.shape[-1]
    x = F.interpolate(img.unsqueeze(0).to(DEV), size=(RESIZE, RESIZE), mode="bilinear", align_corners=False)
    x = x[:, :, PAD:PAD + CROP, PAD:PAD + CROP]
    x = (x - MEAN) / STD
    en, de, _ = model(x)
    amap, _ = cal_anomaly_maps(en, de, CROP)
    canvas = torch.zeros(1, 1, RESIZE, RESIZE, device=DEV)
    canvas[:, :, PAD:PAD + CROP, PAD:PAD + CROP] = amap
    full = F.interpolate(canvas, size=(H, W), mode="bilinear", align_corners=False)
    return full[0, 0].cpu().numpy()


def inp_score(model, img, max_ratio=0.01):
    amap = inp_anomaly_map(model, img)
    flat = np.sort(amap.reshape(-1))[::-1]
    k = max(1, int(flat.size * max_ratio))
    return float(flat[:k].mean())


def _per_image_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return tp / max(tp + fp + fn, 1)


def run_one(name, normals, fit_i, fit_m, test_defs, inp_model):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    ead = det.branches[0]
    en_ = np.array([ead.score(n) for n in normals])
    in_ = np.array([inp_score(inp_model, n) for n in normals])
    imu, isd = in_.mean(), in_.std() + 1e-9

    def z_base(img):
        s = ead.score(img)
        if det._dino is not None:
            return det._dino_fuse(s, det._dino.score(img))
        return (s - en_.mean()) / (en_.std() + 1e-9)

    def z_inp(img):
        return (inp_score(inp_model, img) - imu) / isd

    base_n = np.array([z_base(n) for n in normals])
    base_d = np.array([z_base(d) for d in fit_i])
    thr_base = det.decision_threshold() if det._dino is not None else FewShotAdapter._calibrate(list(base_n), list(base_d))

    inp_n = np.array([z_inp(n) for n in normals])
    inp_d = np.array([z_inp(d) for d in fit_i])
    fused_n = np.maximum(base_n, inp_n)
    fused_d = np.maximum(base_d, inp_d)
    thr_fused = FewShotAdapter._calibrate(list(fused_n), list(fused_d))

    def evaluate(score_fn, thr):
        ious, hits = [], []
        for img, gt in test_defs:
            is_def = score_fn(img) >= thr
            if is_def:
                amap = det.segment(img)
                th = det.pix_thr if det.pix_thr is not None else float(amap.mean() + 3 * amap.std())
                mask = (amap >= th).astype(np.uint8)
                gt_r = (torch.nn.functional.interpolate(
                    torch.from_numpy(gt.astype(np.float32))[None, None],
                    size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
                ious.append(_per_image_iou(mask, gt_r))
                boxes = merge_boxes(map_to_boxes(mask.astype(np.float32), 0.5, min_area_frac=0.0002, close=0),
                                     getattr(det, "box_merge_d", 0))
                hits.append(box_hit(boxes, gt_boxes(gt)) or 0.0)
            else:
                ious.append(0.0); hits.append(0.0)
        return float(np.mean(ious)), float(np.mean(hits))

    base_iou, base_hit = evaluate(z_base, thr_base)
    fused_iou, fused_hit = evaluate(lambda im: max(z_base(im), z_inp(im)), thr_fused)
    print(f"{name:20s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+INP信号 IoU={fused_iou:.3f}/hit={fused_hit:.3f}  Δ(IoU)={fused_iou - base_iou:+.3f}", flush=True)
    return fused_iou - base_iou


def main():
    torch.manual_seed(0)
    print("加载INP-Former官方Few-Shot(k=4) MVTec-AD checkpoint...", flush=True)
    inp_model = build_inp_model()
    print("加载完成", flush=True)

    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("色彩 carpet", lambda: prep_mvtec_color("carpet")[:4]),
        ("色彩 leather", lambda: prep_mvtec_color("leather")[:4]),
        ("色彩 metal_nut", lambda: prep_mvtec_color("metal_nut")[:4]),
        ("色彩 wood", lambda: prep_mvtec_color("wood")[:4]),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        d = run_one(name, *prep(), inp_model)
        names.append(name); deltas.append(d)

    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
              and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()
