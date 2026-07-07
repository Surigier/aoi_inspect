"""DINOv2 图级 co-detector(受控平等融合,治 EAD 漏检 → 提含漏检 IoU)。

洞察:DINOv2 patch 特征图级 AUROC 0.88-1.0(training-free),是强图级检测器,但其强项
不迁移到像素定位(定位融合 α=0)。正确落点=改图级门控:max(z_EAD, z_DINO) 联合阈值,
把 EAD 漏检的缺陷图救回(漏检=定位0分)。

打分器可插拔(SubspaceAD 加固,2026-07):
- "subspace"(默认):对正常 patch 拟合 PCA 主子空间,异常分=落在子空间外的残差能量。
  比最近邻 memory bank 稳——不被个别离群 bank 点带偏(治 cable@640 那种阈值被 fit 离群
  值标崩、全漏的翻车)。training-free、更快(一次 d×k 投影 vs 40000 点匹配)。
- "bank"(AnomalyDINO 式,留作 A/B):L2 归一 patch 余弦最近邻距离。

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
N_COMP = 64                                                  # PCA 子空间维数


class DinoGate:
    """DINOv2 patch 图级异常分(subspace 残差 / bank 最近邻,training-free)。"""

    def __init__(self, device="cuda", mode="bank", n_comp=N_COMP):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mode = mode
        self.n_comp = n_comp
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.bank = None                                     # bank 模式
        self.pca_mean = None; self.subspace = None           # subspace 模式

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
        V = F.normalize(V, dim=1).to(self.device)
        if self.mode == "bank":
            self.bank = V.half()
            return
        # subspace:PCA 主子空间(mean + top-k 正交成分)
        self.pca_mean = V.mean(0)
        Xc = V - self.pca_mean
        k = int(min(self.n_comp, Xc.shape[0] - 1, Xc.shape[1]))
        _, _, Vt = torch.pca_lowrank(Xc, q=k, niter=4)       # Vt:(C,k) 列正交
        self.subspace = Vt[:, :k].contiguous()

    @torch.no_grad()
    def score(self, img):
        q = F.normalize(self._patches(img), dim=1)
        if self.mode == "bank":
            qh = q.half(); d = []
            for i in range(0, qh.shape[0], 2048):
                sim = qh[i:i + 2048] @ self.bank.t()
                d.append(1 - sim.max(dim=1).values.float())
            resid = torch.cat(d).cpu().numpy()
        else:
            xc = q - self.pca_mean                            # (P,C)
            proj = xc @ self.subspace                         # (P,k) 子空间内坐标
            r = (xc * xc).sum(1) - (proj * proj).sum(1)       # 子空间外残差能量(列正交→勾股)
            resid = r.clamp_min(0).sqrt().cpu().numpy()
        k = max(1, int(len(resid) * TOPQ))
        return float(np.sort(resid)[-k:].mean())              # top-1% patch 残差/距离均值


def _bal_acc(scores_d, scores_g, thr):
    rec = np.mean([s >= thr for s in scores_d]) if scores_d else np.nan
    nacc = np.mean([s < thr for s in scores_g]) if scores_g else np.nan
    return (rec + nacc) / 2
