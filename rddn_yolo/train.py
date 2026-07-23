"""YOLO候选框提议器训练(跨12类电子件预训练,学"什么残差是真缺陷/什么是配准噪声"
的通用先验)。自定义Dataset/DataLoader读导出的6通道.npy+YOLO格式.txt标签,但复用
ultralytics官方DetectionModel.loss()(TaskAlignedAssigner+DFL box loss+cls loss,
不是自己重新发明分配/损失逻辑)——只有"怎么读数据"是自定义的,核心训练数学用官方
实现,降低手写bug风险。
用法:PYTHONPATH=. python rddn_yolo/train.py
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from rddn_yolo.model_surgery import make_defect_yolo

ROOT = Path("rddn_yolo/dataset")


class NpyYoloDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = Path(root) / "images" / split
        self.lbl_dir = Path(root) / "labels" / split
        self.files = sorted(self.img_dir.glob("*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = np.load(p).astype(np.float32)                   # (6,H,W)[0,1]
        lbl_path = self.lbl_dir / (p.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in open(lbl_path):
                parts = line.split()
                if len(parts) == 5:
                    boxes.append([float(x) for x in parts])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return torch.from_numpy(img), boxes_t


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    batch_idx, cls, bboxes = [], [], []
    for i, (_, boxes) in enumerate(batch):
        if boxes.shape[0] > 0:
            batch_idx.append(torch.full((boxes.shape[0],), i, dtype=torch.float32))
            cls.append(boxes[:, 0])
            bboxes.append(boxes[:, 1:])
    return {
        "img": imgs,
        "batch_idx": torch.cat(batch_idx) if batch_idx else torch.zeros(0),
        "cls": torch.cat(cls) if cls else torch.zeros(0),
        "bboxes": torch.cat(bboxes) if bboxes else torch.zeros((0, 4)),
    }


def train(epochs=20, batch_size=16, lr=1e-3, out_path="rddn_yolo/defect_yolo.pt",
          root=ROOT, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = make_defect_yolo()
    m.model.to(device).train()
    ds_train = NpyYoloDataset(root, "train")
    ds_val = NpyYoloDataset(root, "val")
    print(f"train样本={len(ds_train)} val样本={len(ds_val)}", flush=True)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    opt = torch.optim.AdamW(m.model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        m.model.train()
        total = 0.0
        for batch in dl_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = m.model.loss(batch)
            loss = loss.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        val_loss = 0.0
        m.model.train()                                        # loss()走训练态forward,eval态输出格式不同
        with torch.no_grad():
            for batch in dl_val:
                batch = {k: v.to(device) for k, v in batch.items()}
                l, _ = m.model.loss(batch)
                val_loss += float(l.sum().item())
        print(f"epoch {epoch:02d}  train_loss={total/max(len(dl_train),1):.4f}  "
              f"val_loss={val_loss/max(len(dl_val),1):.4f}", flush=True)

    torch.save(m.model.state_dict(), out_path)
    print(f"已存: {out_path}", flush=True)
    return m


if __name__ == "__main__":
    train()
