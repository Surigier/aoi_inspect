"""监督分割头:用迁移期标注缺陷掩膜训轻量逐像素逻辑回归(EAD 384通道残差上)。
赛题按分割/检测定位评准确率,且迁移图带标注→用上这监督信号提定位精度。
实验(scripts/run_seg_head.py)验证:像素-AUROC 均值 0.817→0.890,专救弱项。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_to(mask_hw, h, w):
    """(H,W){0,1} numpy → (h,w){0,1},最近邻缩放。"""
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="nearest")[0, 0]
    return (m.numpy() > 0.5).astype(np.uint8)


def _std(m):
    """逐图标准化(减均值除标准差),使不同来源的像素图可比后融合。"""
    return (m - m.mean()) / (m.std() + 1e-6)


def fuse_maps(unsup, sup):
    """无监督异常图 ∪ 监督头图:各自逐图标准化后取 max(保强项又救弱项)。"""
    if sup is None:
        return _std(unsup)
    return np.maximum(_std(unsup), _std(sup))


def map_to_boxes(amap, thr, min_area_frac=0.0008, close=3, max_boxes=20):
    """像素异常图 + 阈值 → 连通域检测框 [(x1,y1,x2,y2,score), ...]。
    赛题评分接受'目标检测定位'。形态学闭运算合并邻近碎块 + 按图面积比过滤小噪点,
    取分最高的 max_boxes 个,避免过碎。"""
    try:
        import cv2
    except Exception:
        return []
    H, W = amap.shape
    binm = (amap >= thr).astype(np.uint8)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    min_area = max(4, int(min_area_frac * H * W))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h), float(amap[y:y + h, x:x + w].max())))
    boxes.sort(key=lambda b: -b[4])
    return boxes[:max_boxes]


class SupervisedSegHead:
    """逐像素特征(C通道)→ 缺陷 logit。fit 用缺陷掩膜+正常负样本;apply 出像素图。
    特征源可插拔(extractor):默认 EAD 残差;生产定位用 WRN50 特征
    (实测 best-IoU 0.263→0.432 +64%,电子件上尤胜 DINOv2)。"""

    def __init__(self, device="cuda", steps=300, lr=0.01, neg_per_img=400, seed=0, n_synth=0,
                 extractor=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.steps, self.lr, self.neg_per_img, self.seed = steps, lr, neg_per_img, seed
        # n_synth 默认0(关):实测合成在同域(赛题场景)可靠掉分-0.02~-0.07,跨域噪声不稳。保留代码opt-in。
        self.n_synth = n_synth
        self.extractor = extractor                         # img(3,H,W)→(C,h,w);None=EAD残差
        self.head = self.mu = self.sd = None
        self.thr = None                                    # fit缺陷掩膜标定的F1最优像素阈值

    @torch.no_grad()
    def _feats(self, det, img):
        f = self.extractor(img) if self.extractor is not None else det.residual_map_large(img)
        C, h, w = f.shape
        return f.reshape(C, -1).t(), (h, w)                # (h*w, C)

    def fit(self, det, defect_imgs, defect_masks, normal_imgs):
        """defect_masks: 每张缺陷的 (H,W){0,1} numpy。normal_imgs:正常图(全负)。"""
        import random as _random
        from .synth import synth_defect
        rng = np.random.RandomState(self.seed)
        srng = _random.Random(self.seed)
        Xs, ys = [], []

        def _add(img, mask_hw):
            feat, (h, w) = self._feats(det, img)
            gt = _mask_to(mask_hw, h, w).ravel()
            pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
            if len(neg) > self.neg_per_img:
                neg = rng.choice(neg, self.neg_per_img, replace=False)
            idx = np.concatenate([pos, neg])
            Xs.append(feat[idx].cpu().numpy()); ys.append(gt[idx])

        for img, mask in zip(defect_imgs, defect_masks):
            _add(img, mask)
        # 反事实合成缺陷(正常→缺陷)扩张分布→泛化(赛题点名合成缺陷),fit时跑不计时
        if self.n_synth and normal_imgs:
            for _ in range(self.n_synth):
                base = normal_imgs[srng.randrange(len(normal_imgs))]
                d_img, d_mask = synth_defect(base, srng)
                _add(d_img, d_mask)
        for img in normal_imgs:
            feat, (h, w) = self._feats(det, img)
            neg = rng.choice(h * w, min(self.neg_per_img, h * w), replace=False)
            Xs.append(feat[neg].cpu().numpy()); ys.append(np.zeros(len(neg), np.uint8))
        X = torch.tensor(np.concatenate(Xs), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=self.device)
        if (y == 1).sum() == 0:                            # 无正样本(掩膜空)→ 不训
            return False
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        self.head = nn.Linear(X.shape[1], 1).to(self.device)
        pos_w = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=self.device)
        opt = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        torch.manual_seed(self.seed)
        for _ in range(self.steps):
            opt.zero_grad(); lossf(self.head(Xn).squeeze(1), y).backward(); opt.step()
        self._calibrate_thr(det, defect_imgs, defect_masks)   # 用fit缺陷掩膜标F1最优阈值
        return True

    def _calibrate_thr(self, det, defect_imgs, defect_masks):
        """在fit缺陷上找最大化F1的阈值(实测校准IoU 0.170→0.269 +58%,弱电子件涨最多)。
        比正常分位阈值好:赛题给了掩膜,监督标定直接对准定位目标。"""
        out_hw = (256, 256)
        S, L = [], []
        for img, mk in zip(defect_imgs, defect_masks):
            S.append(self.map(det, img, out_hw).ravel())
            L.append(_mask_to(mk, out_hw[0], out_hw[1]).ravel())
        s = np.concatenate(S); l = np.concatenate(L)
        order = np.argsort(-s); ls = l[order]; ss = s[order]
        tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
        if P == 0:
            self.thr = None; return
        prec = tp / np.maximum(tp + fp, 1); rec = tp / P
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        self.thr = float(ss[int(np.argmax(f1))])

    @torch.no_grad()
    def map(self, det, img, out_hw):
        """(out_h,out_w) numpy logit 像素图。未训则返回 None。"""
        if self.head is None:
            return None
        feat, (h, w) = self._feats(det, img)
        logit = self.head((feat - self.mu) / self.sd).squeeze(1).reshape(1, 1, h, w)
        amap = F.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        return amap[0, 0].cpu().numpy()
