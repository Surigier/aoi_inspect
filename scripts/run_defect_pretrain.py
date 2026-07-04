"""缺陷分割特化骨干预训练(赛题设计本意:公开集预训练→few-shot迁移;InCTRL/DRA配方)。
在公开缺陷语料(MVTec剔hazelnut/pill + VisA + DAGM + Real-IAD剔pcb/battery)上,
微调 WRN50 layers(1,2) + conv头 做通用"缺陷vs正常"逐像素分割。
产出 models/wrn_defect_l12.pth(骨干权重),迁移时仍冻结(延时/架构不变)。
用法:python scripts/run_defect_pretrain.py [epochs]
"""
import sys
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
SEG_IN = 512; GRID = 128
EXCL_MVTEC = {"hazelnut", "pill"}
EXCL_RIAD = {"pcb", "phone_battery"}
OUT = Path("models/wrn_defect_l12.pth")


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def build_corpus(max_def_per_cat=60, max_norm_per_cat=15):
    """[(img_path_loader, mask(GRID²) or None)];懒加载路径,训练时读。"""
    items = []
    # MVTec(剔除评测类)
    for c in sorted(glob.glob("data/mvtec/*/")):
        cat = Path(c).name
        if cat in EXCL_MVTEC:
            continue
        defs = []
        for dt in sorted(glob.glob(c + "test/*/")):
            dtn = Path(dt).name
            if dtn == "good":
                continue
            for p in sorted(glob.glob(dt + "*.png")):
                mp = GT / cat / "ground_truth" / dtn / (Path(p).stem + "_mask.png")
                defs.append((p, str(mp)))
        random.Random(0).shuffle(defs)
        items += [(p, m) for p, m in defs[:max_def_per_cat]]
        goods = sorted(glob.glob(c + "train/good/*.png"))[:max_norm_per_cat]
        items += [(p, None) for p in goods]
    # DAGM(Test含缺陷,Label目录掩膜)
    for c in sorted(glob.glob("data/dagm/*/")):
        lab = sorted(glob.glob(c + "Test/Label/*_label.PNG"))
        random.Random(0).shuffle(lab)
        for lp in lab[:max_def_per_cat]:
            ip = lp.replace("/Label/", "/").replace("_label.PNG", ".PNG")
            if Path(ip).exists():
                items.append((ip, lp))
        goods = sorted(glob.glob(c + "Train/*.PNG"))[:max_norm_per_cat]
        items += [(p, None) for p in goods]
    # VisA(Images/Anomaly + Masks/Anomaly)
    for c in sorted(glob.glob("data/visa/*/")):
        base = Path(c) / "Data"
        ans = sorted(glob.glob(str(base / "Images/Anomaly/*.JPG")))
        random.Random(0).shuffle(ans)
        for p in ans[:max_def_per_cat]:
            items.append((p, str(base / "Masks/Anomaly" / (Path(p).stem + ".png"))))
        goods = sorted(glob.glob(str(base / "Images/Normal/*.JPG")))[:max_norm_per_cat]
        items += [(p, None) for p in goods]
    # Real-IAD(剔除评测类)
    for jf in sorted(glob.glob(str(RJ / "*.json"))):
        cat = Path(jf).stem
        if cat in EXCL_RIAD:
            continue
        d = json.load(open(jf)); R = RI / cat
        if not R.exists():
            continue
        ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
        random.Random(0).shuffle(ng)
        for x in ng[:max_def_per_cat]:
            items.append((str(R / x["image_path"]), str(R / x["mask_path"])))
        ok = [x for x in d["train"] if x["anomaly_class"] == "OK"][:max_norm_per_cat]
        items += [(str(R / x["image_path"]), None) for x in ok]
    random.Random(0).shuffle(items)
    return items


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("wide_resnet50_2", pretrained=True,
                                    features_only=True, out_indices=(1, 2))
        ch = sum(self.bb.feature_info.channels())
        self.head = nn.Sequential(nn.Conv2d(ch, 64, 1), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(64, 1, 1))

    def feats(self, x):
        fs = self.bb(x)
        size = fs[0].shape[-2:]
        fs = [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fs]
        return torch.cat(fs, dim=1)

    def forward(self, x):
        return self.head(self.feats(x))


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    corpus = build_corpus()
    n_def = sum(1 for _, m in corpus if m)
    print(f"语料: {len(corpus)}图(缺陷{n_def}) epochs={epochs}", flush=True)
    net = Net().to(DEV)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    opt = torch.optim.AdamW([
        {"params": net.bb.parameters(), "lr": 1e-5},       # 骨干小步微调
        {"params": net.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0], device=DEV))
    torch.manual_seed(0)
    step = 0
    for ep in range(epochs):
        random.Random(ep).shuffle(corpus)
        for i in range(0, len(corpus), 8):
            batch = corpus[i:i + 8]
            xs, ys = [], []
            for p, mp in batch:
                try:
                    img = _load_img(p, SEG_IN)
                except Exception:
                    continue
                xs.append(img)
                ys.append(torch.from_numpy(_read(mp, (GRID, GRID)).astype(np.float32)) if mp
                          else torch.zeros(GRID, GRID))
            if not xs:
                continue
            x = torch.stack(xs).to(DEV)
            x = (x - mean) / std
            y = torch.stack(ys).to(DEV)
            opt.zero_grad()
            out = net(x).squeeze(1)
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out[:, None], size=y.shape[-2:], mode="bilinear")[:, 0]
            loss = lossf(out, y)
            loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}", flush=True)
    torch.save(net.bb.state_dict(), OUT)
    print(f"骨干已存 {OUT}", flush=True)


if __name__ == "__main__":
    main()
