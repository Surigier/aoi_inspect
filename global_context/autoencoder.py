"""GCAD风格全局上下文分支(MVTec LOCO论文"Beyond Dents and Scratches"官方baseline
思路的轻量落地):现有EAD(patch级重建误差)和DINO门(patch级最近邻/子空间残差,见
aoi/dino_gate.py)都是逐patch判"这一小块像不像正常"——对"缺件/错位/组合"这类局部
都正常、整体构图错了的逻辑异常天生看不见(背景patch在正常图里到处都有,缺件后
那里还是背景,patch级看不出问题)。GCAD的解法是加一条**真正看整图**的支路:把
整张图压缩过一个瓶颈(bottleneck),用重建误差判"整体构图对不对"——瓶颈维度不够
存下"数量/排布"这类全局信息,只有见过大量正常构图的编码器才能重建准。

两个变体,充分对比,不预设哪个更好:
①PixelAE——忠实原论文:小型卷积自编码器直接在整图(下采样到固定小尺寸)上做重建,
  training-free之外的唯一训练目标就是重建损失,不依赖任何预训练大模型,便宜。
②EmbedAE——用DINOv2的CLS token(整图级语义摘要,ViT自蒸馏训练目标本来就是让CLS
  token代表全局语义)做重建目标,瓶颈MLP自编码器。可以复用aoi/dino_gate.py已有的
  DINOv2前向(同一次forward顺手取CLS token,不必再加一次大模型前向),近乎零增量
  延时;测的是"语义特征本身对全局构图判断有没有额外帮助"(对应用户原问题:是否
  需要更强语义能力),不是"要不要用更大模型"本身。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAE(nn.Module):
    """整图(下采样到64²)卷积自编码器,瓶颈128维——GCAD原论文思路的直接实现。"""

    def __init__(self, in_size=64, bottleneck=128):
        super().__init__()
        self.in_size = in_size
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),      # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),     # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),    # 16->8
            nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU(True),   # 8->4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 128 * 4 * 4), nn.ReLU(True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        z = self.enc(x)
        rec = self.dec(z)
        return rec, x

    def loss(self, x):
        rec, tgt = self.forward(x)
        return F.mse_loss(rec, tgt)

    @torch.no_grad()
    def score(self, x):
        rec, tgt = self.forward(x)
        return float(F.mse_loss(rec, tgt).item())


class EmbedAE(nn.Module):
    """DINOv2 CLS token瓶颈MLP自编码器。cls_dim=384(ViT-S/14)。"""

    def __init__(self, cls_dim=384, bottleneck=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(cls_dim, 128), nn.ReLU(True),
            nn.Linear(128, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(True),
            nn.Linear(128, cls_dim),
        )

    def forward(self, cls_vec):
        z = self.enc(cls_vec)
        rec = self.dec(z)
        return rec

    def loss(self, cls_vec):
        rec = self.forward(cls_vec)
        return F.mse_loss(rec, cls_vec)

    @torch.no_grad()
    def score(self, cls_vec):
        rec = self.forward(cls_vec)
        return float(F.mse_loss(rec, cls_vec).item())


def fit_ae(model, images_or_vecs, steps=300, lr=1e-3, device="cuda", batch=8):
    """通用训练循环:随机批次SGD/Adam,仅在正常样本上最小化重建损失(training-free
    以外的唯一训练目标)。images_or_vecs:list of tensors(图像(3,H,W)或CLS向量(D,))。"""
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    data = torch.stack([t.to(device) for t in images_or_vecs])
    g = torch.Generator().manual_seed(0)
    n = data.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        x = data[idx]
        loss = model.loss(x)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def calibrate_zscore(model, normal_data, is_embed=False):
    """正常样本上算重建误差的mu/sd,供后续z-score融合用(同aoi.competition的
    DINO门标定套路:一致的融合口径,不是另起炉灶)。"""
    device = next(model.parameters()).device
    scores = [model.score(x[None].to(device)) for x in normal_data]
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std() + 1e-9)
