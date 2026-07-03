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


def merge_boxes(boxes, d):
    """近邻(间距<d)框合并为union(碎框合并,fit标定d;实测电子件框命中+0.02~0.04)。"""
    if d <= 0 or len(boxes) <= 1:
        return boxes
    bs = [list(b) for b in boxes]
    changed = True
    while changed and len(bs) > 1:
        changed = False
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                if a[0] - d < b[2] and b[0] - d < a[2] and a[1] - d < b[3] and b[1] - d < a[3]:
                    bs[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                             max(a[4], b[4])]
                    bs.pop(j); changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in bs]


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
    def _fmap(self, det, img):
        """img → (C,h,w) 特征图。"""
        return self.extractor(img) if self.extractor is not None else det.residual_map_large(img)

    @staticmethod
    def _linear_head(C):
        return nn.Conv2d(C, 1, 1)

    @staticmethod
    def _conv_head(C, mid=64):
        """1×1降维+3×3上下文+1×1输出(~10万参)。AD2大图实测+0.119,小图略输线性→fit留出自动选。"""
        return nn.Sequential(nn.Conv2d(C, mid, 1), nn.ReLU(True),
                             nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(True),
                             nn.Conv2d(mid, 1, 1))

    def _train_one(self, head, feats, gts, idxs, pos_w, batch=8):
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            sel = [idxs[i] for i in torch.randperm(len(idxs), generator=g)[:batch]]
            X = torch.stack([feats[i] for i in sel]).to(self.device).float()
            X = (X - self.mu) / self.sd
            Y = torch.stack([gts[i] for i in sel]).to(self.device)
            opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
        head.eval()
        return head

    @torch.no_grad()
    def _hold_iou(self, head, feats, gts, idxs):
        """留出集上:F1最优阈值下逐图IoU均值(选头用)。"""
        logits, labels = [], []
        for i in idxs:
            X = feats[i][None].to(self.device).float()
            logits.append(head((X - self.mu) / self.sd)[0, 0].cpu().numpy())
            labels.append(gts[i].numpy())
        s = np.concatenate([x.ravel() for x in logits]); l = np.concatenate([x.ravel() for x in labels])
        order = np.argsort(-s); ls = l[order]; ss = s[order]
        tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
        f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P, 1e-9)
        thr = float(ss[int(np.argmax(f1))])
        ious = []
        for lo, la in zip(logits, labels):
            pred = lo >= thr
            TP = int((pred & (la == 1)).sum()); FP = int((pred & (la == 0)).sum()); FN = int((~pred & (la == 1)).sum())
            ious.append(TP / max(TP + FP + FN, 1))
        return float(np.mean(ious))

    def fit(self, det, defect_imgs, defect_masks, normal_imgs):
        """全图训练(实测优于像素采样:pill+0.10/pcb+0.04)+ 双头留出自动选(线性vs卷积)。
        defect_masks: 每张缺陷 (H,W){0,1} numpy。normal_imgs:正常图(全负)。"""
        import random as _random
        from .synth import synth_defect
        srng = _random.Random(self.seed)
        items = list(zip(defect_imgs, defect_masks))
        if self.n_synth and normal_imgs:
            for _ in range(self.n_synth):
                items.append(synth_defect(normal_imgs[srng.randrange(len(normal_imgs))], srng))
        feats, gts, is_def = [], [], []
        grid = None
        with torch.no_grad():
            for img, mk in items:
                f = self._fmap(det, img)
                if grid is None:
                    grid = f.shape[-2:]
                if f.shape[-2:] != grid:
                    continue                                # 变尺寸特征图跳过(生产extractor为定尺寸)
                feats.append(f.half().cpu())
                gts.append(torch.from_numpy(_mask_to(mk, grid[0], grid[1]).astype(np.float32)))
                is_def.append(True)
            for img in normal_imgs[:20]:
                f = self._fmap(det, img)
                if f.shape[-2:] != grid:
                    continue
                feats.append(f.half().cpu()); gts.append(torch.zeros(grid)); is_def.append(False)
        if not feats or not any(g.sum() > 0 for g in gts):
            return False
        stack_mean = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0)
        stack_sd = torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6
        self.mu = stack_mean.view(1, -1, 1, 1).to(self.device)
        self.sd = stack_sd.view(1, -1, 1, 1).to(self.device)
        pos = sum(float(g.sum()) for g in gts); neg = sum(g.numel() for g in gts) - pos
        pos_w = torch.tensor([neg / max(pos, 1)], device=self.device)
        C = feats[0].shape[0]
        def_idx = [i for i, d in enumerate(is_def) if d]
        all_idx = list(range(len(feats)))
        # 留出选头(缺陷≥8才选;否则直接线性)
        torch.manual_seed(self.seed)
        if len(def_idx) >= 8:
            hold = def_idx[::4]
            train_idx = [i for i in all_idx if i not in set(hold)]
            lin = self._train_one(self._linear_head(C).to(self.device), feats, gts, train_idx, pos_w)
            cnv = self._train_one(self._conv_head(C).to(self.device), feats, gts, train_idx, pos_w)
            iou_l = self._hold_iou(lin, feats, gts, hold)
            iou_c = self._hold_iou(cnv, feats, gts, hold)
            use_conv = iou_c > iou_l + 0.01                 # 卷积要赢出margin才用(稳定性偏向线性)
            self.head_kind = "conv" if use_conv else "linear"
        else:
            use_conv = False; self.head_kind = "linear"
        # 全量重训胜者
        head = (self._conv_head(C) if use_conv else self._linear_head(C)).to(self.device)
        self.head = self._train_one(head, feats, gts, all_idx, pos_w)
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
        f = self._fmap(det, img)[None].to(self.device).float()   # (1,C,h,w)
        logit = self.head((f - self.mu) / self.sd)
        amap = F.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        return amap[0, 0].cpu().numpy()
