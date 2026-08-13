"""在4图拼接的2500×2500数据上跑一遍真实生产det.fit_fewshot+locate,看看真实赛制
分辨率(可能的拼接场景)下会不会有意外状况(崩溃/延时爆炸/精度异常下降),这是为
竞赛环境做的针对性铺垫测试,不是新机制验证。

用法:PYTHONPATH=. python tile_2500/run_tiled_probe.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import merge_boxes, map_to_boxes
from global_context.eval_global_branch import gt_boxes, box_hit
from tile_2500.prep_tiled import prep_tiled


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def run_one(name, normals, fit_i, fit_m, test_defs, n_test=8):
    print(f"{name}: 拼接后 normals={len(normals)} fit_i={len(fit_i)} test={len(test_defs)},"
          f"图像shape={normals[0].shape}", flush=True)
    det = CompetitionLargeDetector()
    t0 = time.time()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"{name}: fit_fewshot耗时={time.time()-t0:.1f}s  seg_head训成功={det.seg_head.head is not None}",
          flush=True)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    ious, hits, lat = [], [], []
    for i, (img, gt) in enumerate(test_defs[:n_test]):
        t1 = time.time()
        o = det.locate(img)
        dt = time.time() - t1
        lat.append(dt)
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0)
            print(f"  [{i}] is_defect={o['is_defect']} 未判缺陷 延时={dt*1000:.0f}ms", flush=True)
            continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        iou = _per_image_iou(mask, gt_r)
        # gt_boxes()要求GT和预测框在同一坐标系;o["boxes"]来自det.segment()固定输出
        # 的256×256(seg_eval_hw),原生gt是2500²拼接出来的——之前忘了缩放,导致
        # hit永远算成0(坐标尺度差~10倍,预测框和GT框根本不可能相交)。用已经缩放好
        # 的gt_r(和mask同分辨率)算gt_boxes,而不是原生分辨率的gt。
        hit = box_hit(o["boxes"], gt_boxes(gt_r)) or 0.0
        ious.append(iou); hits.append(hit)
        print(f"  [{i}] is_defect={o['is_defect']} iou={iou:.3f} hit={hit:.3f} 延时={dt*1000:.0f}ms "
              f"boxes={len(o['boxes'])}", flush=True)

    print(f"{name}: 均值IoU={np.mean(ious):.3f} 均值hit={np.mean(hits):.3f} "
          f"延时p50={np.median(lat)*1000:.0f}ms p90={np.percentile(lat,90)*1000:.0f}ms", flush=True)
    return dict(iou=np.mean(ious), hit=np.mean(hits), lat_p50=np.median(lat))


def main():
    torch.manual_seed(0)
    print("=== hazelnut(拼接2500²) ===", flush=True)
    run_one("拼接 hazelnut", *prep_tiled("mvtec", "hazelnut", ["crack", "cut", "hole"]))

    print("\n=== pcb(拼接2500²) ===", flush=True)
    run_one("拼接 pcb", *prep_tiled("realiad", "pcb"))


if __name__ == "__main__":
    main()
