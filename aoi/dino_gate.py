"""DINOv2 图级 co-detector(受控平等融合,治 EAD 漏检 → 提含漏检 IoU)。

洞察:DINOv2 patch 记忆库图级 AUROC 0.88-1.0(training-free),是强图级检测器,但其强
项不迁移到像素定位(定位融合 α=0)。正确落点=改图级门控:max(z_EAD, z_DINO) 联合阈值,
把 EAD 漏检的缺陷图救回(漏检=定位0分)。实测 battery 平衡acc +0.125、hazelnut +0.022,
且召回与正常acc 同升(≠灰区补检的单向翻转必加误报)。

安全:A/B 半交叉验证,仅当融合门 B 半平衡acc 不劣于 EAD-only 才逐类启用,否则回退纯EAD。
延时:ViT-S/14 @518 每图一次前向 ~25-40ms(仅图级门,不进逐块定位)。
"""
import numpy as np
import torch
import torch.nn.functional as F
import timm

DINO_SZ = 518
BANK_MAX = 40000
TOPQ = 0.01


class DinoGate:
    """DINOv2 patch 记忆库图级分(AnomalyDINO 式,余弦最近邻,training-free)。"""

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.bank = None

    @torch.no_grad()
    def _patches(self, img):
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = x.to(self.device)
        x = F.interpolate(x, size=(DINO_SZ, DINO_SZ), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:, :]  # (1,N,C)
        return t[0].float()

    def build(self, normals):
        vs = [self._patches(n).cpu() for n in normals]
        V = torch.cat(vs)
        g = torch.Generator().manual_seed(0)
        if V.shape[0] > BANK_MAX:
            V = V[torch.randperm(V.shape[0], generator=g)[:BANK_MAX]]
        self.bank = F.normalize(V, dim=1).half().to(self.device)

    @torch.no_grad()
    def score(self, img):
        q = F.normalize(self._patches(img), dim=1).half()
        d = []
        for i in range(0, q.shape[0], 2048):
            sim = q[i:i + 2048] @ self.bank.t()
            d.append(1 - sim.max(dim=1).values.float())
        dm = torch.cat(d).cpu().numpy()
        k = max(1, int(len(dm) * TOPQ))
        return float(np.sort(dm)[-k:].mean())  # top-1% patch 距离均值


def _bal_acc(scores_d, scores_g, thr):
    rec = np.mean([s >= thr for s in scores_d]) if scores_d else np.nan
    nacc = np.mean([s < thr for s in scores_g]) if scores_g else np.nan
    return (rec + nacc) / 2
