"""轻量判别头:在骨干特征上做逻辑回归,用现场正常+缺陷监督训练。
反馈累积样本 → 重训 → 权重(模型参数)更新,实现赛题"动态调整模型参数"。"""
import torch
import torch.nn as nn


class DiscriminativeHead:
    def __init__(self, epochs: int = 200, lr: float = 0.05, device: str = "cpu"):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.lin = None

    def fit(self, normal_feats: torch.Tensor, defect_feats: torch.Tensor):
        """normal_feats/defect_feats: (N, D) 图像级特征。每次调用重训(权重重置)。"""
        assert len(defect_feats) >= 1 and len(normal_feats) >= 1, "判别头需正常与缺陷各至少一例"
        x = torch.cat([normal_feats, defect_feats]).float().to(self.device)
        y = torch.cat([torch.zeros(len(normal_feats)),
                       torch.ones(len(defect_feats))]).to(self.device)
        torch.manual_seed(0)                          # 重训可复现
        self.lin = nn.Linear(x.shape[1], 1).to(self.device)
        opt = torch.optim.Adam(self.lin.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.lin(x).squeeze(1), y)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def score(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (M, D) -> (M,) 缺陷概率 [0,1]。"""
        assert self.lin is not None, "score 前须先 fit"
        return torch.sigmoid(self.lin(feat.float().to(self.device)).squeeze(-1))
