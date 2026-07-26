"""AHL(Anomaly Heterogeneity Learning,CVPR2024,出题方唯一点名的参考文献)思路
适配到我们的`SupervisedSegHead`训练过程——这是赛题指定文献里方法论上最贴合我们
场景的一条:少量标注异常(我们是15~30张)+需要泛化到训练时没见过的异常(我们是
隐藏测试域的手机部件),原论文管这个叫"open-set supervised anomaly detection"。

核心思路(按论文改造,忠实核心机制,简化了部分细节——见下方"简化说明"):
①把fit期正常图在WRN特征空间聚成C类(k-means)
②每次随机取一个正常簇 + 随机70%的标注缺陷,组成一个"异态代理数据集"的support;
  剩下30%缺陷做该代理集的query(对这个代理集是"没见过的")。重复T次得到T个代理集。
③每个代理集的support上各自训一个"基础头"(和生产SupervisedSegHead同样的
  linear+conv双头集成架构)——这些基础头只是训练过程的中间产物,不用于推理。
④在全部T个代理集的query集(每个query对训练该代理集用的基础头是"没见过的",
  拼起来构成一个模拟"开放集泛化"的训练信号)上,协同训练**一个最终统一头**——
  这才是真正用于推理的模型。
⑤推理只用④训出来的统一头,**和现有生产seg_head调用方式完全一样,不增加任何
  推理延时**(论文原文:"During inference, only the unified model g is used")。

简化说明(诚实标注,不是偷懒回避,是避免在验证核心机制前引入太多变量):
- 原论文用CutMix/CutPaste/DRAEM给每个代理集额外注入伪异常提高异态性;这里第一版
  不加(我们独立试过CutPaste单独作为增广已判负,不想让"伪异常生成"这个已知有
  风险的环节混进"异态代理集"这个新机制的验证信号里,先确认核心机制本身有没有
  用,伪异常增强留作后续可选消融)。
- 原论文用BiLSTM学习每个基础头的动态重要性权重;这里第一版做**无加权**的协同
  训练(所有query样本等权重),更简单、更容易归因,如果核心思路有效再考虑加权重。
- T默认7(论文默认值),C默认3(论文默认值)。

【已验证,判负封存,见ahl_seghead/eval_ahl.py】3类摸底(breakfast_box逻辑异常/
pushpins逻辑异常/cable生产回归检查)3/3全负且方向一致(ΔIoU=-0.037/-0.039/-0.015,
median=-0.037 mean=-0.030),不像GCAD实验那样有类目间正负交替的噪声信号——这次是
干净的、方向一致的负结果,没有扩大到9类目普查的必要(信号已经够清楚,不是"看运气")。

**最可能的根因**:我们的fit缺陷集本来就极小(LOCO约15张,cable约10~15张)。这套
机制的核心动作是把这个已经很小的集合再拆成T=7个"support(70%)/query(30%)"子集
训练——原论文场景(DevNet/DRA分类式打分网络,论文实验设置下异常样本量/评测方式
和我们不同)下这样切片能提升泛化,但对我们这种**逐像素分割**任务、且标注样本
本来就少到individual instance都数得过来的场景,再切片等于让每个子集看到的独特
缺陷实例更少——"人为制造开放集泛化信号"这个收益,没能盖过"可用监督信号被进一步
稀释"这个代价。**这不代表AHL的思路在更大样本量场景下没用,是我们"30张缺陷"这个
量级本身可能太小,撑不起论文原本假设的"充分切片仍有信号"这个前提。**默认不接入
competition.py,代码留opt-in研究件。
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to


def _pool(feat):
    """(C,h,w) -> (C,) 全局平均池化,供聚类用。"""
    return feat.mean(dim=(1, 2))


def kmeans_np(X, k, iters=50, seed=0):
    """极简k-means(不引入sklearn依赖)。X:(N,D) numpy。返回长度N的簇标签。"""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = X[labels == c].mean(0)
    return labels


def build_surrogate_splits(normal_feats, defect_feats, defect_masks, grid_hw,
                           T=7, C=3, support_frac=0.7, seed=0):
    """返回T个(support_idx, query_idx, normal_cluster_feats)。idx索引进defect_feats/masks。"""
    pooled = np.stack([_pool(f).cpu().numpy() for f in normal_feats])
    labels = kmeans_np(pooled, C, seed=seed)
    clusters = [[normal_feats[i] for i in range(len(normal_feats)) if labels[i] == c]
               for c in range(C)]
    clusters = [c for c in clusters if len(c) > 0]
    n_def = len(defect_feats)
    rng = random.Random(seed)
    splits = []
    for t in range(T):
        idx = list(range(n_def))
        rng.shuffle(idx)
        n_sup = max(1, int(n_def * support_frac))
        sup_idx, qry_idx = idx[:n_sup], idx[n_sup:]
        if not qry_idx:                                    # 太少缺陷时至少留1张做query
            qry_idx = [sup_idx.pop()]
        cluster = clusters[t % len(clusters)]
        splits.append((sup_idx, qry_idx, cluster))
    return splits


def _train_head(feats_pos, masks_pos, feats_neg, grid_hw, steps=150, lr=5e-3, seed=0, device="cuda"):
    """和生产SupervisedSegHead._train_one同架构(linear+conv双头集成),独立训练循环
    (不复用production的fit()整体流程,只借它的头架构,方便按support/query自由拼数据)。"""
    gh, gw = grid_hw
    all_feats = feats_pos + feats_neg
    all_gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)) for m in masks_pos] + \
              [torch.zeros(gh, gw) for _ in feats_neg]
    C = all_feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in all_feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in all_feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    pos = sum(float(g.sum()) for g in all_gts); neg = sum(g.numel() for g in all_gts) - pos
    pos_w = torch.tensor([neg / max(pos, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    torch.manual_seed(seed)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    n = len(all_feats)
    for _ in range(steps):
        i = int(torch.randint(0, n, (1,), generator=g).item())
        X = all_feats[i][None].to(device).float()
        X = (X - mu) / sd
        Y = all_gts[i][None].to(device)
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    return head, mu, sd


def fit_ahl_unified(normal_feats, defect_feats, defect_masks, grid_hw,
                    T=7, C=3, support_frac=0.7, base_steps=100, unified_steps=250,
                    seed=0, device="cuda"):
    """完整AHL适配流程,返回(unified_head, mu, sd)——推理只用这三样,和production
    SupervisedSegHead的self.head/self.mu/self.sd等价,可以直接换进map()逻辑。"""
    if len(defect_feats) < 4:
        return None, None, None
    splits = build_surrogate_splits(normal_feats, defect_feats, defect_masks, grid_hw,
                                    T=T, C=C, support_frac=support_frac, seed=seed)
    # ③ 每个代理集训一个基础头(中间产物,只用于本函数内部,不返回)
    query_pool_feats, query_pool_masks = [], []
    for t, (sup_idx, qry_idx, cluster_normals) in enumerate(splits):
        sup_feats = [defect_feats[i] for i in sup_idx]
        sup_masks = [defect_masks[i] for i in sup_idx]
        _base_head, _mu, _sd = _train_head(sup_feats, sup_masks, cluster_normals[:10], grid_hw,
                                           steps=base_steps, seed=seed + t, device=device)
        # ④ 该代理集的query样本(对刚训的基础头是"没见过的")汇入统一训练池
        for i in qry_idx:
            query_pool_feats.append(defect_feats[i])
            query_pool_masks.append(defect_masks[i])
    # 统一头在全部query池上训练(每个样本至少来自一个"它没参与训练"的基础头的query集)
    all_normals = [f for _, _, cluster in splits for f in cluster[:5]]
    unified_head, mu, sd = _train_head(query_pool_feats, query_pool_masks, all_normals, grid_hw,
                                       steps=unified_steps, lr=5e-3, seed=seed, device=device)
    return unified_head, mu, sd
