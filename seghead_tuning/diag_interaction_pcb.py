"""查pcb在"seg_head门控+GCAD-EmbedAE"组合下比"只有seg_head门控"更差(-0.032 vs
-0.016)的真正原因:是GCAD改变了图级检测判定(该抓的没抓到/多抓了不该抓的),还是
纯粹是掩膜定位质量变差了。逐图对比两种设置下的is_defect判定和per-image IoU。

用法:PYTHONPATH=. python seghead_tuning/diag_interaction_pcb.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import _mask_to
from seghead_tuning.gated_train import pick_gated_head, _test_iou_with_head
from seghead_tuning.probe_aggressive_train import _per_image_iou
from global_context.eval_global_branch import prep_realiad, fit_global_branches, DinoCLS, gt_boxes, box_hit


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs = prep_realiad("pcb")
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    extractor = det.seg_head.extractor

    picked = pick_gated_head("生产:pcb", extractor, fit_i, fit_m)
    base_cfg, gated_cfg = picked
    h_g, mu_g, sd_g, thr_g, winner = gated_cfg
    print(f"门控选中: {winner}", flush=True)

    # ① 只有seg_head门控(不接GCAD):把门控头写入det,用原生EAD+DINO判定图级
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = h_g, mu_g, sd_g
    det.seg_head.thr = thr_g; det.seg_head.rams = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DinoCLS(device=device)
    fns = fit_global_branches(det, normals, fit_i, dino, ae_steps=900, ae_lr=3e-3)
    z_base_fn, thr_base_gate = fns["base"]   # 只有EAD+DINO,不接GCAD
    z_emb_fn, thr_emb_gate = fns["emb"]      # 接了GCAD-EmbedAE

    print(f"图级阈值: 只EAD+DINO={thr_base_gate:.3f}  接GCAD后={thr_emb_gate:.3f}", flush=True)

    print(f"{'idx':4s} {'只seg_head门控':>22s} {'+GCAD':>22s}", flush=True)
    rows = []
    for i, (img, gt) in enumerate(test_defs):
        s_base = z_base_fn(img); s_emb = z_emb_fn(img)
        is_def_base = s_base >= thr_base_gate
        is_def_emb = s_emb >= thr_emb_gate

        def _iou_if_flagged(is_def):
            if not is_def:
                return 0.0
            amap = det.segment(img)
            th = det.pix_thr if det.pix_thr is not None else float(amap.mean() + 3 * amap.std())
            mask = (amap >= th).astype(np.uint8)
            gt_r = (torch.nn.functional.interpolate(
                torch.from_numpy(gt.astype(np.float32))[None, None],
                size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
            return _per_image_iou(mask, gt_r)

        iou_base = _iou_if_flagged(is_def_base)
        iou_emb = _iou_if_flagged(is_def_emb)
        rows.append((is_def_base, iou_base, is_def_emb, iou_emb))
        flip = " <-- 判定翻转" if is_def_base != is_def_emb else ""
        print(f"{i:4d}  flag={str(is_def_base):5s} iou={iou_base:.3f}      "
              f"flag={str(is_def_emb):5s} iou={iou_emb:.3f}{flip}", flush=True)

    n_flip_to_miss = sum(1 for b, _, e, _ in rows if b and not e)   # 加了GCAD反而漏检了
    n_flip_to_catch = sum(1 for b, _, e, _ in rows if not b and e)  # 加了GCAD反而多抓到了
    mean_base = np.mean([r[1] for r in rows]); mean_emb = np.mean([r[3] for r in rows])
    print(f"\n判定翻转统计: 加GCAD后新漏检={n_flip_to_miss}张  新多抓到={n_flip_to_catch}张", flush=True)
    print(f"均值IoU: 只seg_head门控={mean_base:.3f}  +GCAD={mean_emb:.3f}  Δ={mean_emb-mean_base:+.3f}", flush=True)


if __name__ == "__main__":
    main()
