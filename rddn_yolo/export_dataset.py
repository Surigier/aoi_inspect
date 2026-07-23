"""数据集导出:GT掩膜→YOLO格式框标注,跨12个Real-IAD电子/手机件类目组装训练语料。
每样本存(a)6通道.npy(RGB+差异通道,640²,float16省盘)(b)YOLO格式.txt标签(单类
"defect",归一化x_center,y_center,w,h)。正样本来自真实缺陷图(mask→框),负样本
来自normal-normal配对(空标签,教网络认"这只是配准/光照噪声不是缺陷")。
每类先取合理规模(200缺陷+200负样本)保证本次能跑完,不是一次性导出全部~82000张——
后续如果这条路验证有效,可以再加量。
用法:PYTHONPATH=. python rddn_yolo/export_dataset.py
"""
import json
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from aoi.imageio import load_fast
from rddn_yolo.diff_channels import build_6ch

RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
OUT = Path("rddn_yolo/dataset")
SIZE = 640
CATS = ["phone_battery", "pcb", "sim_card_set", "usb", "usb_adaptor", "switch",
        "button_battery", "terminalblock", "transistor1", "regulator", "audiojack", "end_cap"]
N_PER_CAT_DEFECT = 200
N_PER_CAT_NEG = 200


def _gray_key(img, size=32):
    arr = (img.mean(0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(arr, (size, size)).astype(np.float32)


def _resize_chw(x, size):
    """x:(C,H,W) numpy float -> (C,size,size),bilinear。"""
    t = torch.from_numpy(x)[None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].numpy()


def _mask_to_yolo_boxes(mask_hw, min_area=4):
    """mask(H,W){0,1} -> [(cx,cy,w,h)]归一化到[0,1](单类,不分缺陷类型)。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_hw.astype(np.uint8), connectivity=8)
    H, W = mask_hw.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append(((x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return out


def _save_sample(idx_name, split, ch6_640, boxes):
    img_dir = OUT / "images" / split; lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    np.save(img_dir / f"{idx_name}.npy", ch6_640.astype(np.float16))
    with open(lbl_dir / f"{idx_name}.txt", "w") as f:
        for cx, cy, w, h in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    random.seed(0)
    n_pos_total = n_neg_total = 0
    for cat in CATS:
        d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
        tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]
        random.Random(0).shuffle(tok)
        pool_n = min(60, len(tok))
        templates = [load_fast(R / x["image_path"]) for x in tok[:pool_n]]
        tmpl_keys = np.stack([_gray_key(t) for t in templates])
        extra_normals = tok[pool_n:pool_n + N_PER_CAT_NEG]        # 负样本本体(和template不重复)

        ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
        random.Random(1).shuffle(ng)
        n_pos = n_neg = 0
        for x in ng[:N_PER_CAT_DEFECT * 2]:                       # 多取一些,跳过无效mask后仍够数
            if n_pos >= N_PER_CAT_DEFECT:
                break
            mp = x.get("mask_path")
            if mp is None or not (R / mp).exists():
                continue
            img = load_fast(R / x["image_path"])
            gt_native = (np.array(Image.open(R / mp).convert("L")) > 0).astype(np.uint8)
            if gt_native.sum() < 4:
                continue
            qk = _gray_key(img)
            i = int(np.argmin(((tmpl_keys - qk) ** 2).mean(axis=(1, 2))))
            ch6 = build_6ch(img, templates[i])                    # (6,h,w) 原生尺度
            ch6_640 = _resize_chw(ch6, SIZE)
            gt_640 = cv2.resize(gt_native, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            boxes = _mask_to_yolo_boxes(gt_640)
            if not boxes:
                continue
            split = "val" if n_pos % 5 == 0 else "train"          # 按图分80/20,不按裁块(防泄漏)
            _save_sample(f"{cat}_pos_{n_pos:04d}", split, ch6_640, boxes)
            n_pos += 1
        for x in extra_normals[:N_PER_CAT_NEG]:
            if n_neg >= N_PER_CAT_NEG:
                break
            img = load_fast(R / x["image_path"])
            j = random.Random(n_neg).randrange(len(templates))
            ch6 = build_6ch(img, templates[j])                    # normal vs normal:差异应只是噪声
            ch6_640 = _resize_chw(ch6, SIZE)
            split = "val" if n_neg % 5 == 0 else "train"
            _save_sample(f"{cat}_neg_{n_neg:04d}", split, ch6_640, [])   # 空标签
            n_neg += 1
        print(f"{cat:14s} 正样本={n_pos} 负样本={n_neg}", flush=True)
        n_pos_total += n_pos; n_neg_total += n_neg

    yaml_path = OUT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\n"
                f"nc: 1\nnames: ['defect']\n")
    print(f"\n=== 总计 === 正样本={n_pos_total} 负样本={n_neg_total}  写到 {OUT}/  "
          f"dataset.yaml={yaml_path}", flush=True)


if __name__ == "__main__":
    main()
