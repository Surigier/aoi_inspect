"""连通域计数的便宜版:不用SAM(everything模式7秒/张,已判负),复用crop_cascade.py
已有的ECC对齐模板残差(Lab色差+梯度差→连通域,training-free,原生就是给独立
crop-head候选用的),把"残差最强候选的分数"当第三个图级检测信号,融合进
max(z_EAD, z_DINO, z_residual)——和当年给cable加DINO同一个融合模式。

动机:缺件/错位这类逻辑异常,test图和(ECC对齐后的)正常模板做像素级残差比对时,
"少了个东西"的位置会留下明显的色差/梯度差空洞——这是一个和EAD/DINO本质不同的
信号来源(基于模板对齐残差,不是patch级重建误差或最近邻距离),延时量级接近
box-prompted SAM(~20-40ms),不是SAM everything模式的秒级开销。

成功判据:图级检测层面,加了这个信号后,gated IoU(含漏检)在LOCO逻辑异常类目上
median(Δ)>=0.005且min(Δ)>=-0.01(和今天margin判定口径一致),同时cable/pcb这类
回归检查不能明显变差。

【已验证,判负】5类(breakfast_box/pushpins/screw_bag逻辑+cable/pcb回归检查):
ΔIoU=[0.000,-0.020,+0.032,0.000,0.000],median=0.000 mean=+0.003 min=-0.020,
不通过。breakfast_box/cable/pcb三类是0.000(残差信号从没比EAD+DINO更高过,
根本没触发过任何判决变化,不是"触发了但没用"),screw_bag唯一一次真正生效
且是正的(+0.032),pushpins生效但是负的(-0.020)。**结论:这个残差信号大部分
时候太弱,压不过EAD/DINO,少数触发时正负各半,没有稳定收益。** 默认不接入
competition.py,代码留opt-in研究件。

用法:PYTHONPATH=. python unified_student/probe_residual_gate.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.fewshot import FewShotAdapter
from aoi.crop_cascade import _lab_grad_candidates
from aoi.seg_head import map_to_boxes, merge_boxes
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _residual_score(det, img):
    """ECC对齐模板残差最强候选的分数,当作一个标量图级信号(和EAD/DINO同量纲用法)。"""
    native = img if img.dim() == 3 else img[0]
    ref = det._ref_bank.aligned_ref(native)
    cands = _lab_grad_candidates(native, ref, topk=1)
    return float(cands[0][4]) if cands else 0.0


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def run_one(name, normals, fit_i, fit_m, test_defs):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)          # 同今天已知的GPU负载下DINO被砍风险,强制补标定

    ead = det.branches[0]
    en = np.array([ead.score(n) for n in normals]); ed = np.array([ead.score(d) for d in fit_i])
    rn = np.array([_residual_score(det, n) for n in normals]); rd = np.array([_residual_score(det, d) for d in fit_i])
    emu, esd = en.mean(), en.std() + 1e-9
    rmu, rsd = rn.mean(), rn.std() + 1e-9

    def z_base(img):
        s = ead.score(img)
        if det._dino is not None:
            return det._dino_fuse(s, det._dino.score(img))
        return (s - emu) / esd

    def z_resid(img):
        r = _residual_score(det, img)
        return (r - rmu) / rsd

    base_n = np.array([z_base(n) for n in normals]); base_d = np.array([z_base(d) for d in fit_i])
    thr_base = det.decision_threshold() if det._dino is not None else FewShotAdapter._calibrate(list(base_n), list(base_d))

    fused_n = np.maximum(base_n, [z_resid(n) for n in normals])
    fused_d = np.maximum(base_d, [z_resid(d) for d in fit_i])
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
    fused_iou, fused_hit = evaluate(lambda im: max(z_base(im), z_resid(im)), thr_fused)
    print(f"{name:28s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+残差信号 IoU={fused_iou:.3f}/hit={fused_hit:.3f}  Δ(IoU)={fused_iou-base_iou:+.3f}", flush=True)
    return dict(base=(base_iou, base_hit), fused=(fused_iou, fused_hit))


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
        ("logical:screw_bag", lambda: prep_loco("screw_bag", "logical_anomalies")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("生产:pcb(回归检查)", lambda: prep_realiad("pcb")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        deltas.append(row["fused"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'✅通过' if passed else '❌不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()
