"""监督分割头:用迁移期标注缺陷掩膜训轻量逐像素逻辑回归(EAD 384通道残差上)。
赛题按分割/检测定位评准确率,且迁移图带标注→用上这监督信号提定位精度。
实验(scripts/run_seg_head.py)验证:像素-AUROC 均值 0.817→0.890,专救弱项。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_to(mask_hw, h, w):
    """(H,W){0,1} numpy → (h,w){0,1},最近邻缩放(供阈值/评估等要硬标签的场景用)。
    训练场景请用 _mask_to_soft(area-preserving,避免极小掩膜被采样点漏掉)。"""
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="nearest")[0, 0]
    return (m.numpy() > 0.5).astype(np.uint8)


def _mask_to_soft(mask_hw, h, w):
    """(H,W){0,1} numpy → (h,w)[0,1] 连续占用率,area-preserving下采样(等价平均池化)。
    最近邻缩放在下采样比例大时(如2500²→128²,~20×)会让占几个像素的微小缺陷(PCB划痕/
    小点)整体漏采样归零;area插值保留每个输出格子里正像素的真实占比,训练直接用这个
    软标签(soft occupancy target)而非阈值到硬0/1,信息不丢失。"""
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="area")[0, 0]
    return m.numpy().astype(np.float32)


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


def _gt_boxes_np(mask, min_area=4):
    """(h,w){0,1} → GT连通域框 [(x1,y1,x2,y2), ...](供OOF阈值搜索用,自包含不依赖scripts/)。"""
    try:
        import cv2
    except Exception:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= min_area]


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _box_hit_rate(pred_masks, gt_masks, thr_iou=0.5):
    """逐图预测掩膜(阈值二值化后)vs GT掩膜,连通域框@IoU0.5命中率(按GT框计,自包含版)。"""
    tot = hit = 0
    for pm, gm in zip(pred_masks, gt_masks):
        preds = _gt_boxes_np(pm)
        for g in _gt_boxes_np(gm):
            tot += 1
            hit += any(_box_iou(p, g) >= thr_iou for p in preds)
    return hit / max(tot, 1)


def _per_image_iou(pred_masks, gt_masks):
    ious = []
    for pm, gm in zip(pred_masks, gt_masks):
        p = pm.astype(bool); g = gm.astype(bool)
        TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
        ious.append(TP / max(TP + FP + FN, 1))
    return float(np.mean(ious)) if ious else 0.0


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


class _Ensemble(nn.Module):
    """双头logit平均(WBF思想:组合优于选择;消30掩膜小留出的选择方差)。"""
    def __init__(self, *heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)
    def forward(self, x):
        out = None
        for h in self.heads:
            o = h(x)
            out = o if out is None else out + o
        return out / len(self.heads)


class _RamsCorr(nn.Module):
    """RAMS-R 残差注意力多尺度修正支(锚定已训双头基线):逐尺度1×1降维 → 逐像素逐尺度
    softmax注意力加权融合 → zero-conv头(末层零初始化→出发点≡基线,只学修正)。
    诊断实测(run_rams_diag.py,3种子):从零训RAMS优化不稳定判负,锚定版AD2 3/4类
    +0.013~+0.033(std≤0.022)、最差-0.009噪声内 → fit留出门控逐类启用,零回退。"""

    def __init__(self, chans, d=48):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv2d(c, d, 1) for c in chans])
        self.att = nn.Conv2d(d * len(chans), len(chans), 1)
        self.head = nn.Sequential(nn.Conv2d(d, 64, 1), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, scales):
        r = [red(s) for red, s in zip(self.red, scales)]
        w = torch.softmax(self.att(torch.cat(r, 1)), dim=1)
        return self.head(sum(w[:, i:i + 1] * r[i] for i in range(len(r))))


class SupervisedSegHead:
    """逐像素特征(C通道)→ 缺陷 logit。fit 用缺陷掩膜+正常负样本;apply 出像素图。
    特征源可插拔(extractor):默认 EAD 残差;生产定位用 WRN50 特征
    (实测 best-IoU 0.263→0.432 +64%,电子件上尤胜 DINOv2)。"""

    def __init__(self, device="cuda", steps=300, lr=0.01, neg_per_img=400, seed=0, n_synth=0,
                 extractor=None, rams_extractor=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.steps, self.lr, self.neg_per_img, self.seed = steps, lr, neg_per_img, seed
        # n_synth 默认0(关):实测合成在同域(赛题场景)可靠掉分-0.02~-0.07,跨域噪声不稳。保留代码opt-in。
        self.n_synth = n_synth
        self.extractor = extractor                         # img(3,H,W)→(C,h,w);None=EAD残差
        self.rams_extractor = rams_extractor               # img→[多尺度(C_i,h,w)];RAMS-R修正支用
        self.head = self.mu = self.sd = None
        # 5折OOF标定的3个候选阈值(_oof_calibrate_thr):self.thr默认=thr_iou(逐图平均IoU最优,
        # 赛题评分主指标);thr_boxhit/thr_f1供按官方最终输出类型切换(见run_pareto_scan等脚本)。
        self.thr = self.thr_iou = self.thr_boxhit = self.thr_f1 = None
        self.oof_maps = None                               # {原defect下标: OOF预测图256²}(下游门控无偏base)
        self.rams = None                                   # RAMS-R修正支(fit留出门控启用)
        self._rams_stats = None

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

    def _train_one(self, head, feats, gts, idxs, pos_w=None, batch=8, seed=None):
        """BCE+SoftDice,按图等权平均(不是按批次全部像素平均)。此前纯BCE对batch内全部
        像素取mean→大缺陷图(像素多)天然主导梯度,小缺陷(如PCB划痕)被淹没;现在每张图
        各自算一次loss(pos_weight按该图自己的正负比例定,不用全局pos_w)再等权平均。
        gts为soft occupancy target(_mask_to_soft,area-preserving,支持连续标签)。"""
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        seed = self.seed if seed is None else seed
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        for _ in range(self.steps):
            sel = [idxs[i] for i in torch.randperm(len(idxs), generator=g)[:batch]]
            X = torch.stack([feats[i] for i in sel]).to(self.device).float()
            X = (X - self.mu) / self.sd
            Y = torch.stack([gts[i] for i in sel]).to(self.device)          # (B,h,w) soft[0,1]
            logit = head(X).squeeze(1)
            prob = torch.sigmoid(logit)
            losses = []
            for b in range(logit.shape[0]):
                yb, lb, pb = Y[b], logit[b], prob[b]
                pos_frac = yb.mean().clamp(min=1e-4)
                pw = ((1 - pos_frac) / pos_frac).clamp(max=500)
                bce = F.binary_cross_entropy_with_logits(lb, yb, pos_weight=pw)
                inter = (pb * yb).sum()
                dice = 1 - (2 * inter + 1) / (pb.sum() + yb.sum() + 1)
                losses.append(bce + dice)
            loss = torch.stack(losses).mean()                               # 按图等权,非按像素
            opt.zero_grad(); loss.backward(); opt.step()
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
        """全图训练(实测优于像素采样:pill+0.10/pcb+0.04)+ N头bagging集成(线性+3×3卷积+
        不同种子轻头,logit平均;WRN特征只算一次,多头推理开销可忽略)。
        defect_masks: 每张缺陷 (H,W){0,1} numpy。normal_imgs:正常图(全负)。
        训练目标改为soft occupancy(_mask_to_soft,area-preserving)+ BCE按图等权+SoftDice,
        避免大缺陷图(像素多)主导梯度、小缺陷(PCB划痕)被淹没。"""
        import random as _random
        from .synth import synth_defect
        srng = _random.Random(self.seed)
        n_real = len(defect_imgs)
        items = list(zip(defect_imgs, defect_masks))
        if self.n_synth and normal_imgs:
            for _ in range(self.n_synth):
                items.append(synth_defect(normal_imgs[srng.randrange(len(normal_imgs))], srng))
        feats, gts, is_def = [], [], []
        real_pos, real_orig_idx = [], []           # 真实缺陷(非合成)在feats里的位置+对应defect_masks原始下标
        grid = None
        with torch.no_grad():
            for idx, (img, mk) in enumerate(items):
                f = self._fmap(det, img)
                if grid is None:
                    grid = f.shape[-2:]
                if f.shape[-2:] != grid:
                    continue                                # 变尺寸特征图跳过(生产extractor为定尺寸)
                feats.append(f.half().cpu())
                gts.append(torch.from_numpy(_mask_to_soft(mk, grid[0], grid[1])))
                is_def.append(True)
                if idx < n_real:
                    real_pos.append(len(feats) - 1); real_orig_idx.append(idx)
            normal_pos = []
            for img in normal_imgs[:20]:
                f = self._fmap(det, img)
                if f.shape[-2:] != grid:
                    continue
                feats.append(f.half().cpu()); gts.append(torch.zeros(grid)); is_def.append(False)
                normal_pos.append(len(feats) - 1)
        if not feats or not any(g.sum() > 0 for g in gts):
            return False
        stack_mean = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0)
        stack_sd = torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6
        self.mu = stack_mean.view(1, -1, 1, 1).to(self.device)
        self.sd = stack_sd.view(1, -1, 1, 1).to(self.device)
        C = feats[0].shape[0]
        all_idx = list(range(len(feats)))
        # bagging集成(4头:线性+3×3卷积×3种子)替代双头:各自随机批次采样带来的方差被平均掉,
        # 推理只是多几次1×1/3×3小卷积,相对WRN骨干前向可忽略不计。
        torch.manual_seed(self.seed)
        heads = [self._train_one(self._linear_head(C).to(self.device), feats, gts, all_idx, seed=self.seed)]
        for k in range(3):
            heads.append(self._train_one(self._conv_head(C).to(self.device), feats, gts, all_idx,
                                         seed=self.seed + 1 + k))
        self.head = _Ensemble(*heads)
        self.head_kind = f"bagging{len(heads)}"
        self._fit_rams(det, defect_imgs, defect_masks, normal_imgs)  # RAMS-R修正支(留出门控)
        self._oof_calibrate_thr(det, feats, gts, real_pos, real_orig_idx, normal_pos,
                                defect_masks, C)
        return True

    def _oof_calibrate_thr(self, det, feats, gts, real_pos, real_orig_idx, normal_pos,
                           defect_masks_native, C, k=5):
        """5折OOF阈值标定(替代原pooled-pixel F1——那个口径下大缺陷图像素多天然主导阈值,
        实测正好伤PCB这类小缺陷)。每折训一个抛弃式轻头(不用全量bagging省时间),预测其
        未见过的验证折,汇总所有折OOF预测后优化3个候选阈值:①逐图平均IoU最优(评分主指标,
        默认self.thr用它)②框命中@0.5最优③F1最优(旧口径,保留供对比/按官方输出类型切换)。
        真实缺陷不足8张时退化为2折。"""
        n = len(real_pos)
        if n < 4:
            self.thr = self.thr_iou = self.thr_boxhit = self.thr_f1 = None
            return
        kk = k if n >= 2 * k else max(2, n // 3)
        order = list(range(n))
        import random as _r
        _r.Random(self.seed).shuffle(order)
        folds = [order[i::kk] for i in range(kk)]

        out_hw = (256, 256)
        oof_amap, oof_gt = {}, {}
        for fi in range(kk):
            val_p = folds[fi]
            tr_p = [q for j, f in enumerate(folds) if j != fi for q in f]
            tr_feat_idx = [real_pos[q] for q in tr_p] + normal_pos
            if len(tr_feat_idx) < 2:
                continue
            head = self._train_one(self._conv_head(C).to(self.device), feats, gts, tr_feat_idx,
                                   seed=self.seed + 200 + fi)
            with torch.no_grad():
                for q in val_p:
                    X = feats[real_pos[q]][None].to(self.device).float()
                    logit = head((X - self.mu) / self.sd)
                    up = F.interpolate(logit, size=out_hw, mode="bilinear",
                                       align_corners=False)[0, 0].cpu().numpy()
                    oof_amap[q] = up
                    oof_gt[q] = _mask_to(defect_masks_native[real_orig_idx[q]], out_hw[0], out_hw[1])
        if not oof_amap:
            self.thr = self.thr_iou = self.thr_boxhit = self.thr_f1 = None
            return
        keys = sorted(oof_amap.keys())
        amaps = [oof_amap[k_] for k_ in keys]; gtsN = [oof_gt[k_] for k_ in keys]
        all_vals = np.concatenate([a.ravel() for a in amaps])
        cands = sorted(set(float(np.percentile(all_vals, q)) for q in range(1, 100)))
        if not cands:
            self.thr = self.thr_iou = self.thr_boxhit = self.thr_f1 = None
            return

        def _best(metric_fn):
            vals = [(t, metric_fn(t)) for t in cands]
            return max(vals, key=lambda x: x[1])[0]

        def iou_at(t):
            return _per_image_iou([(a >= t) for a in amaps], gtsN)

        def boxhit_at(t):
            return _box_hit_rate([(a >= t).astype(np.uint8) for a in amaps], gtsN)

        def f1_at(t):
            l = np.concatenate([g.ravel() for g in gtsN])
            pred = all_vals >= t
            TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
            prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
            return 2 * prec * rec / max(prec + rec, 1e-9)

        self.thr_iou = _best(iou_at)
        self.thr_boxhit = _best(boxhit_at)
        self.thr_f1 = _best(f1_at)
        self.thr = self.thr_iou                  # 默认口径=逐图平均IoU(赛题评分主指标)
        # 留存OOF预测图(键=原defect_masks下标):下游门控(如组件图)评估"某机制对seg的边际
        # 增益"必须用未见折预测当base——用self.map()会因这些图正是训练图而base过拟合地好,
        # 系统性低估边际增益(组件图门控实测在juice_bottle上-0.097 vs test真值+0.080的教训)。
        self.oof_maps = {real_orig_idx[q]: oof_amap[q] for q in keys}

    def _calibrate_thr(self, det, defect_imgs, defect_masks):
        """[已弃用,由_oof_calibrate_thr取代]保留供opt-in/对比用:pooled-pixel F1阈值,
        大缺陷图像素多天然主导,实测会伤PCB这类小缺陷占比高的图。"""
        out_hw = (256, 256)
        S, L = [], []
        for img, mk in zip(defect_imgs, defect_masks):
            S.append(self.map(det, img, out_hw).ravel())
            L.append(_mask_to(mk, out_hw[0], out_hw[1]).ravel())
        s = np.concatenate(S); l = np.concatenate(L)
        order = np.argsort(-s); ls = l[order]; ss = s[order]
        tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
        if P == 0:
            return None
        prec = tp / np.maximum(tp + fp, 1); rec = tp / P
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        return float(ss[int(np.argmax(f1))])

    def _fit_rams(self, det, defect_imgs, defect_masks, normal_imgs, steps=400, margin=0.005):
        """RAMS-R 残差修正支训练+留出门控:锚定已训双头(零初始化→出发点≡基线),只学修正;
        留出集(每4取1)IoU 无增益(≤margin)→ 不启用,零回退。见 _RamsCorr 注释与 run_rams_diag。"""
        self.rams = None
        if self.rams_extractor is None or len(defect_imgs) < 8:
            return
        hold = list(range(0, len(defect_imgs), 4))
        tr = [i for i in range(len(defect_imgs)) if i not in set(hold)]
        # 缓存训练集:base logit(特征格,双头冻结)+ 多尺度特征 + gt
        tr_imgs = [defect_imgs[i] for i in tr] + list(normal_imgs[:15])
        tr_mks = [defect_masks[i] for i in tr] + [None] * min(15, len(normal_imgs))
        B, S, G = [], [], []
        with torch.no_grad():
            for img, mk in zip(tr_imgs, tr_mks):
                f = self._fmap(det, img)[None].to(self.device).float()
                B.append(self.head((f - self.mu) / self.sd).cpu())
                sc = self.rams_extractor(img)
                S.append([s.half().cpu() for s in sc])
                g = sc[0].shape[-2:]
                G.append(torch.from_numpy(_mask_to(mk, g[0], g[1]).astype(np.float32))
                         if mk is not None else torch.zeros(g))
        if B[0].shape[-2:] != S[0][0].shape[-2:]:
            return                                            # base与修正支网格不一致(如变尺寸)→跳过
        nsc = len(S[0])
        mus, sds = [], []
        for i in range(nsc):
            A = torch.stack([s[i].float() for s in S])
            mus.append(A.mean(dim=(0, 2, 3), keepdim=True).to(self.device))
            sds.append((A.std(dim=(0, 2, 3), keepdim=True) + 1e-6).to(self.device))
        self._rams_stats = (mus, sds)
        torch.manual_seed(self.seed)
        corr = _RamsCorr([s.shape[0] for s in S[0]]).to(self.device)
        pos = sum(float(g.sum()) for g in G); neg = sum(g.numel() for g in G) - pos
        pw = torch.tensor([neg / max(pos, 1)], device=self.device)
        opt = torch.optim.Adam(corr.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        gg = torch.Generator().manual_seed(self.seed)
        N = len(S)
        for _ in range(steps):
            sel = torch.randperm(N, generator=gg)[:8].tolist()
            scales = [torch.stack([S[j][i].float() for j in sel]).to(self.device) for i in range(nsc)]
            scales = [(scales[i] - mus[i]) / sds[i] for i in range(nsc)]
            base = torch.cat([B[j] for j in sel]).to(self.device)
            gt = torch.stack([G[j] for j in sel]).to(self.device)
            opt.zero_grad()
            lossf((base + corr(scales)).squeeze(1), gt).backward()
            opt.step()
        corr.eval()
        # 留出门控:base vs base+corr 各自F1阈值下的逐图IoU
        h_imgs = [defect_imgs[i] for i in hold]; h_mks = [defect_masks[i] for i in hold]

        def hold_iou(use_corr):
            self.rams = corr if use_corr else None
            out_hw = (256, 256)
            Sm = [self.map(det, im, out_hw) for im in h_imgs]
            Lm = [_mask_to(mk, out_hw[0], out_hw[1]) for mk in h_mks]
            s = np.concatenate([x.ravel() for x in Sm]); l = np.concatenate([x.ravel() for x in Lm])
            order = np.argsort(-s); ls = l[order]; ss = s[order]
            tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
            f1 = 2 * (tp / np.maximum(tp + fp, 1)) * (tp / P) / np.maximum(
                (tp / np.maximum(tp + fp, 1)) + (tp / P), 1e-9)
            thr = float(ss[int(np.argmax(f1))])
            ious = []
            for smap, lm in zip(Sm, Lm):
                pred = smap >= thr
                TP = int((pred & (lm == 1)).sum()); FP = int((pred & (lm == 0)).sum())
                FN = int((~pred & (lm == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(ious))

        base_iou = hold_iou(False)
        corr_iou = hold_iou(True)
        if corr_iou > base_iou + margin:
            self.rams = corr
            self.rams_gain = corr_iou - base_iou              # 留出增益(诊断/日志用)
        else:
            self.rams = None

    @torch.no_grad()
    def map(self, det, img, out_hw):
        """(out_h,out_w) numpy logit 像素图。未训则返回 None。"""
        if self.head is None:
            return None
        f = self._fmap(det, img)[None].to(self.device).float()   # (1,C,h,w)
        logit = self.head((f - self.mu) / self.sd)
        if self.rams is not None:                                # RAMS-R残差修正(留出门控启用时)
            mus, sds = self._rams_stats
            sc = self.rams_extractor(img)
            sc = [((sc[i][None].to(self.device).float() - mus[i]) / sds[i]) for i in range(len(sc))]
            if sc[0].shape[-2:] == logit.shape[-2:]:
                logit = logit + self.rams(sc)
        amap = F.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        return amap[0, 0].cpu().numpy()
