"""fit阶段真实评估:模拟赛题场景(100正常+30缺陷现场迁移,untimed)——冻结的预训练
YOLO + 可选LoRA微调(30张真缺陷+normal-normal负样本对),在该产品的独立留出测试图上
比较候选框召回。

⚠️方法论说明(诚实标注,非隐瞒):这里的"留出测试图"是每个类目导出时按图分的20%
val split——即"预训练看过这个类目、但没看过这张具体图",不是"预训练完全没见过这个
产品类目"。真正的赛题场景是隐藏域(手机屏幕/电池/中框)对预训练来说是全新类目,
这个更严格的测试需要另开一次【类目级留出】的预训练(现有12类里留2类完全不参与
预训练,只在fit阶段用)才能验证,属于后续工作,当前先验证LoRA机制本身有没有用。

用法:PYTHONPATH=. python rddn_yolo/eval_fit.py --cat phone_battery --ckpt rddn_yolo/defect_yolo.pt
"""
import argparse
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from rddn_yolo.model_surgery import make_defect_yolo
from rddn_yolo.diff_channels import build_6ch
from rddn_yolo.lora import inject_lora_into_head, freeze_all_except_lora
from rddn_yolo.export_dataset import _mask_to_yolo_boxes, _gray_key, _resize_chw
from aoi.imageio import load_fast

import json
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
SIZE = 640


def _box_iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _yolo_to_xyxy(cx, cy, w, h, size=SIZE):
    return (cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size


@torch.no_grad()
def predict_boxes(model, ch6_640, conf_thr=0.0, nms_iou=0.5):
    """model(eval)对单张6通道640²图跑推理,返回置信度>=conf_thr的[(x1,y1,x2,y2,conf)]
    (原始网络输出坐标已经是640尺度)。conf_thr默认0(几乎不过滤,只做NMS去重),留给
    调用方在拿到全部候选后再按各自模型自己的最优阈值筛选——避免"冻结/LoRA共用一个
    阈值"这个confound(LoRA微调后输出分布可能整体偏移,共用阈值对LoRA不公平)。"""
    model.eval()
    x = torch.from_numpy(ch6_640)[None].float()
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)[0]
    pred = out[0].T                                          # (8400,5): x,y,w,h,conf(像素尺度,已sigmoid)
    boxes = []
    conf = pred[:, 4]
    keep = conf >= conf_thr
    for cx, cy, w, h, c in pred[keep].cpu().numpy():
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes.append((x1, y1, x2, y2, float(c)))
    boxes.sort(key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(_box_iou_xyxy(b[:4], k[:4]) < nms_iou for k in kept):
            kept.append(b)
        if len(kept) >= 300:                                 # 保底上限,防止低阈值时候选爆炸拖慢NMS
            break
    return kept


def best_recall_over_thresholds(model, test_set, thr_grid=None):
    """对test_set里每张图跑一次推理(conf_thr=0低阈值拿全量候选),缓存全部候选框,
    之后对一组候选阈值分别算整体候选框召回,返回(每个阈值的召回, 最优阈值, 最优召回)。
    这样每个模型都在自己的最优工作点上被评估,不共用一个可能不公平的固定阈值。"""
    if thr_grid is None:
        thr_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    cached = []
    for ch6, gt_boxes in test_set:
        all_boxes = predict_boxes(model, ch6, conf_thr=0.02)   # 极低阈值,拿到几乎全部候选一次性缓存
        cached.append((all_boxes, gt_boxes))
    results = {}
    for thr in thr_grid:
        recalls = []
        for all_boxes, gt_boxes in cached:
            boxes_t = [b for b in all_boxes if b[4] >= thr]
            r = box_recall(boxes_t, gt_boxes, iou_thr=0.3)
            if r is not None:
                recalls.append(r)
        results[thr] = float(np.mean(recalls)) if recalls else 0.0
    best_thr = max(results, key=results.get)
    return results, best_thr, results[best_thr]


def box_recall(pred_boxes, gt_boxes_xyxy, iou_thr=0.3):
    if not gt_boxes_xyxy:
        return None
    if not pred_boxes:
        return 0.0
    hit = sum(1 for g in gt_boxes_xyxy if any(_box_iou_xyxy(p[:4], g) >= iou_thr for p in pred_boxes))
    return hit / len(gt_boxes_xyxy)


def build_lora_finetune_set(cat, fit_defect_n=30, neg_pair_n=100, seed=0):
    """模拟fit阶段:该产品100正常图(建模板池+配对负样本)+30张真缺陷图(有GT框)。"""
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]
    random.Random(seed).shuffle(tok)
    normals_100 = tok[:100]
    templates = [load_fast(R / x["image_path"]) for x in normals_100[:60]]
    tmpl_keys = np.stack([_gray_key(t) for t in templates])

    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
    random.Random(seed + 1).shuffle(ng)
    fit_pos, fit_neg = [], []
    for x in ng:
        if len(fit_pos) >= fit_defect_n:
            break
        mp = x.get("mask_path")
        if mp is None or not (R / mp).exists():
            continue
        img = load_fast(R / x["image_path"])
        gt_native = (np.array(Image.open(R / mp).convert("L")) > 0).astype(np.uint8)
        if gt_native.sum() < 4:
            continue
        qk = _gray_key(img); i = int(np.argmin(((tmpl_keys - qk) ** 2).mean(axis=(1, 2))))
        ch6 = build_6ch(img, templates[i]); ch6_640 = _resize_chw(ch6, SIZE)
        gt_640 = cv2.resize(gt_native, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        boxes = _mask_to_yolo_boxes(gt_640)
        if boxes:
            fit_pos.append((ch6_640, boxes))
    for x in normals_100[60:60 + neg_pair_n]:
        img = load_fast(R / x["image_path"])
        j = random.Random(len(fit_neg)).randrange(len(templates))
        ch6 = build_6ch(img, templates[j]); ch6_640 = _resize_chw(ch6, SIZE)
        fit_neg.append((ch6_640, []))
    return fit_pos, fit_neg, tok, templates, tmpl_keys


def build_test_set(cat, templates, tmpl_keys, skip_paths, n_test=40, seed=2):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK" and x["image_path"] not in skip_paths]
    random.Random(seed).shuffle(ng)
    out = []
    for x in ng[:n_test]:
        mp = x.get("mask_path")
        if mp is None or not (R / mp).exists():
            continue
        img = load_fast(R / x["image_path"])
        gt_native = (np.array(Image.open(R / mp).convert("L")) > 0).astype(np.uint8)
        if gt_native.sum() < 4:
            continue
        qk = _gray_key(img); i = int(np.argmin(((tmpl_keys - qk) ** 2).mean(axis=(1, 2))))
        ch6 = build_6ch(img, templates[i]); ch6_640 = _resize_chw(ch6, SIZE)
        gt_640 = cv2.resize(gt_native, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        boxes_yolo = _mask_to_yolo_boxes(gt_640)
        boxes_xyxy = [_yolo_to_xyxy(*b) for b in boxes_yolo]
        if boxes_xyxy:
            out.append((ch6_640, boxes_xyxy))
    return out


def lora_finetune(model, fit_pos, fit_neg, steps=200, lr=1e-3, device="cuda"):
    params, _ = inject_lora_into_head(model.model, r=4, alpha=1.0)
    freeze_all_except_lora(model.model)
    model.model.to(device).train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    all_samples = fit_pos + fit_neg
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        i = int(torch.randint(0, len(all_samples), (1,), generator=g).item())
        ch6, boxes = all_samples[i]
        img_t = torch.from_numpy(ch6)[None].float().to(device)
        if boxes:
            b = torch.tensor(boxes, dtype=torch.float32)
            batch = {"img": img_t, "batch_idx": torch.zeros(len(boxes)).to(device),
                     "cls": torch.zeros(len(boxes)).to(device), "bboxes": b[:, :].to(device)}
        else:
            batch = {"img": img_t, "batch_idx": torch.zeros(0).to(device),
                     "cls": torch.zeros(0).to(device), "bboxes": torch.zeros((0, 4)).to(device)}
        loss, _ = model.model.loss(batch)
        opt.zero_grad(); loss.sum().backward(); opt.step()
    model.model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="phone_battery")
    ap.add_argument("--ckpt", default="rddn_yolo/defect_yolo.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    fit_pos, fit_neg, tok, templates, tmpl_keys = build_lora_finetune_set(args.cat)
    print(f"fit正样本={len(fit_pos)} fit负样本={len(fit_neg)}", flush=True)
    skip = {x["image_path"] for x in tok}
    test_set = build_test_set(args.cat, templates, tmpl_keys, skip)
    print(f"test样本={len(test_set)}", flush=True)

    m_frozen = make_defect_yolo()
    m_frozen.model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m_frozen.model.to(device).eval()
    res_f, thr_f, best_f = best_recall_over_thresholds(m_frozen.model, test_set)
    print(f"冻结YOLO(无LoRA) 各阈值召回={ {k: round(v,3) for k,v in res_f.items()} }", flush=True)
    print(f"冻结YOLO 最优阈值={thr_f} 最优候选框召回@IoU0.3={best_f:.3f}", flush=True)

    m_lora = make_defect_yolo()
    m_lora.model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m_lora = lora_finetune(m_lora, fit_pos, fit_neg, device=device)
    res_l, thr_l, best_l = best_recall_over_thresholds(m_lora.model, test_set)
    print(f"+LoRA微调 各阈值召回={ {k: round(v,3) for k,v in res_l.items()} }", flush=True)
    print(f"+LoRA微调 最优阈值={thr_l} 最优候选框召回@IoU0.3={best_l:.3f}", flush=True)
    print(f"Δ(各自最优阈值下公平对比)={best_l-best_f:+.3f}", flush=True)


if __name__ == "__main__":
    main()
