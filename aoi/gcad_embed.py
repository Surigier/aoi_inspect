"""GCAD风格全局语义分支——**已接入生产又立刻回退,默认关(2026-08-13)**,记录留档。

现有EAD(patch级重建误差)和DINO门(patch级最近邻/子空间残差)都是逐patch判"这一
小块像不像正常",对"缺件/错位/组合"这类局部都正常、整体构图错了的逻辑异常天生
看不见。这里加一条真正看整图的支路:把DINOv2的CLS token(整图级语义摘要)过一个
瓶颈MLP自编码器,重建误差判"整体构图对不对"。复用aoi/dino_gate.py的DinoGate同一次
DINOv2前向顺手取的CLS token(gate.last_cls,零增量前向),不再加载第二个DINOv2实例。

【真实教训】研究阶段验证(global_context/eval_aggressive.py、eval_emb_prod5.py)只在
test_defs(缺陷图)上算IoU/hit,**从没用正常图测过假阳性率**,得出"9类/5类不需门控
直接过严格margin判据,min=0.000"的结论——这个判据口径本身有方法论盲区,只测了
"该抓的有没有抓到"这一侧,完全没测"正常图会不会被误报"。真上生产`scripts/
run_scorecard.py`(含正常图的完整图级acc)一测:图级acc从0.902崩到0.703(-0.199),
IoU/框命中只有+0.006~+0.022的小幅提升,完全不能抵消这个代价——OR门(base判"正常"
时才查EmbedAE独立阈值)只要阈值稍微松一点,在"正常图占多数"的真实测试集上,误报
的绝对数量就会很大,不是只看缺陷图的min=0.000能测出来的。

判定接入方式(OR门)本身的设计没问题:base(EAD+DINO)该抓的用现有判定原样保留,
只有base判"正常"时才额外查EmbedAE是否独立超过自己的阈值——比之前试过的max()+
联合重标定阈值(会把base原有边界跟着抬高,反而漏检,见diag_interaction_pcb.py)
更安全。**问题出在EmbedAE自己的独立阈值(FewShotAdapter._calibrate标定)太松,
没有专门针对"正常图假阳性率"这个目标去标定**——只用了缺陷分离阈值的常规标法,
没有加"正常图误报率必须低于X%"这类约束。

真正要用,必须先补上：①用held-out正常图测独立分支的假阳性率(不能只看缺陷图召回)
②给独立阈值加保守margin或专门按控制假阳性率的方式标定,不能直接套用EAD/DINO那套
"分离正常/缺陷"的常规阈值标定逻辑。default=False,代码留opt-in研究件。

用法:CompetitionLargeDetector(gcad_embed=False)默认关(见aoi/competition.py)。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS, LR = 900, 3e-3  # 激进配置,见上方docstring


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
        return self.dec(z)

    def loss(self, cls_vec):
        return F.mse_loss(self.forward(cls_vec), cls_vec)

    @torch.no_grad()
    def score(self, cls_vec):
        return float(F.mse_loss(self.forward(cls_vec), cls_vec).item())


def fit_embed_ae(cls_vecs, device="cuda", steps=STEPS, lr=LR, batch=8):
    """cls_vecs: list of (384,) CLS token张量(仅正常图)。"""
    model = EmbedAE().to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    data = torch.stack([t.to(device) for t in cls_vecs])
    g = torch.Generator().manual_seed(0)
    n = data.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        opt.zero_grad(); model.loss(data[idx]).backward(); opt.step()
    model.eval()
    return model


def calibrate_zscore(model, normal_cls_vecs):
    """正常样本CLS token重建误差的mu/sd,标定口径同DINO门(z-score融合)。"""
    device = next(model.parameters()).device
    scores = np.array([model.score(v[None].to(device)) for v in normal_cls_vecs])
    return float(scores.mean()), float(scores.std() + 1e-9)
