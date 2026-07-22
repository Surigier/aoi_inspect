"""DCP-SFR启发的边界残差头(CVPR2026 Defect Cue-Preserved Structural Feature Refinement,
只借鉴思想不照搬整模型):深层特征逐步丢失细小缺陷信号,故用浅层edge cue修正分割
logit边界——同我们既有证据("WRN浅层优于DINO、PCB信号随下采样消失")完全一致的方向。

输入(拼接):
- base_logit:seg_head当前输出(1通道)
- 原图灰度Sobel梯度 + Lab a通道Sobel梯度(2通道,边缘/色彩边界线索,training-free)
- WRN layer1浅层特征(256通道,_wrn_feats输出的前256通道就是layer1,同一次前向
  的"免费"切片,不额外提特征)

网络:1x1降维 → 两层depthwise3x3+pointwise1x1 → 1x1零初始化输出。
refined_logit = base_logit + correction(correction起点恒为0,起点≡raw,不会变差)。

训练损失:BCE + SoftDice + BoundaryDice(GT掩膜边界环上的Dice,专注边界精度)。

门控:k折OOF(不是单次留出——RAMS-R"8张留出门控噪声漏判强类"生产判负的教训,今天
seg_head/component_graph两次单次/小样本门控踩坑也印证了这点)比较raw vs refined
的OOF held-out IoU,净正过margin才启用。不强制过SAM(SAM有自己独立的逐区域门控,
在refine()之后照常生效,两者解耦——避免RAMS-R"SAM下游重塑边界冲掉raw增益"重演)。
opt-in,默认关,fit留出验证净正才启用。

【真实数据验证结果,2026-07-20,run_boundary_refine_ab.py】DCP-SFR目标场景(微小
缺陷/边界丢失)pcb/battery+AD2 sheet_metal三类整体负:pcb门控正确拦截(test集强制
开Δ纯定位=-0.075/Δ框=-0.200),battery同样门控正确拦截(Δ=-0.069/-0.050);唯一
门控判"开"的sheet_metal(fit留出估gain=+0.074)在独立test集只有-0.012/+0.043——
又一次印证"fit侧正增益估计不可靠"(今天第三次撞到这堵墙)。均值Δ纯定位=-0.052、
Δ框=-0.069。结论:DCP-SFR的边界残差思想在本项目当前形态下未见真实收益,默认关闭
是对的,暂不作为生产候选,代码留作opt-in研究件(若要继续投入,方向可能是重新审视
输入特征/loss设计,而非门控本身——门控这次至少在2/3类上判断方向正确)。"""
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from .seg_head import _mask_to, _per_image_iou


def _edge_feat(img_chw, hw):
    """img(3,H,W)[0,1] cpu/gpu tensor -> (2,H,W) numpy: 灰度Sobel幅值 + Lab-a通道Sobel幅值,
    各自归一化到[0,1]。纯图像处理,training-free,无额外网络前向。"""
    arr = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    arr = cv2.resize(arr, (hw[1], hw[0]))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2); gmag = gmag / (gmag.max() + 1e-6)
    lab_a = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[..., 1].astype(np.float32)
    ax = cv2.Sobel(lab_a, cv2.CV_32F, 1, 0, ksize=3); ay = cv2.Sobel(lab_a, cv2.CV_32F, 0, 1, ksize=3)
    amag = np.sqrt(ax ** 2 + ay ** 2); amag = amag / (amag.max() + 1e-6)
    return np.stack([gmag, amag], axis=0)


def _boundary_ring(mask_hw, width=3):
    """GT掩膜边界环(膨胀-腐蚀,BoundaryDice专注区域)。mask_hw:(H,W){0,1} uint8。"""
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask_hw, k, iterations=width)
    ero = cv2.erode(mask_hw, k, iterations=width)
    return (dil - ero).astype(np.uint8)


def _soft_dice(logit, target, eps=1.0):
    p = torch.sigmoid(logit)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


class _RefineNet(nn.Module):
    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, hidden, 1)
        self.dw1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw1 = nn.Conv2d(hidden, hidden, 1)
        self.dw2 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw2 = nn.Conv2d(hidden, hidden, 1)
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 零初始化:起点≡raw

    def forward(self, x):
        h = F.relu(self.reduce(x))
        h = F.relu(self.pw1(self.dw1(h)))
        h = F.relu(self.pw2(self.dw2(h)))
        return self.out(h)


class BoundaryRefiner:
    def __init__(self, device="cuda", hidden=32, steps=300, lr=5e-3, seed=0):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hidden = hidden
        self.steps = steps
        self.lr = lr
        self.seed = seed
        self.net = None
        self.enabled = False
        self.gain = None

    def _build_input(self, det, img, base_logit_hw):
        """(C,H,W) tensor:[base_logit(1)+edge(2)+layer1浅层特征(256)]。"""
        H, W = base_logit_hw.shape
        native = img if img.dim() == 3 else img[0]
        f = det._wrn_feats(native)                            # (768,gh,gw),前256=layer1
        f1 = F.interpolate(f[:256][None], size=(H, W), mode="bilinear", align_corners=False)[0]
        edge = torch.from_numpy(_edge_feat(native, (H, W))).to(f1.device).float()
        base = torch.from_numpy(base_logit_hw).to(f1.device).float()[None]
        return torch.cat([base, edge, f1], dim=0)

    def _train_one(self, det, imgs, base_logits, masks_native, steps, seed):
        torch.manual_seed(seed)
        C = 259
        net = _RefineNet(C, self.hidden).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        cache = []
        for img, bl, mk in zip(imgs, base_logits, masks_native):
            H, W = bl.shape
            x = self._build_input(det, img, bl)
            gt = torch.from_numpy(_mask_to(mk, H, W).astype(np.float32)).to(self.device)
            ring = torch.from_numpy(_boundary_ring(_mask_to(mk, H, W)).astype(np.float32)).to(self.device)
            cache.append((x, torch.from_numpy(bl).to(self.device).float(), gt, ring))
        if not cache:
            return net
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            i = int(torch.randint(0, len(cache), (1,), generator=g).item())
            x, bl, gt, ring = cache[i]
            corr = net(x[None])[0, 0]
            logit = bl + corr
            bce = F.binary_cross_entropy_with_logits(logit, gt)
            dice = _soft_dice(logit, gt)
            bd = _soft_dice(logit * ring, gt * ring) if ring.sum() > 0 else torch.tensor(0.0, device=self.device)
            loss = bce + dice + 0.5 * bd
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        return net

    @torch.no_grad()
    def _apply(self, net, det, img, base_logit_hw):
        x = self._build_input(det, img, base_logit_hw)
        corr = net(x[None])[0, 0].cpu().numpy()
        return base_logit_hw + corr

    def fit(self, det, defect_imgs, defect_masks, k=4, margin=0.01):
        """defect_imgs原生尺度,defect_masks原生{0,1}掩膜。k折OOF比较raw vs refined,
        净正过margin才启用;否则zero-footprint(self.net保持None)。"""
        self.enabled = False
        self.net = None
        if not _HAS_CV2 or len(defect_imgs) < 8:
            return
        base_logits = [det.segment(img) for img in defect_imgs]     # 复用segment()已算的seg_head输出
        n = len(defect_imgs)
        kk = min(k, max(2, n // 4))
        order = list(range(n)); _random.Random(self.seed).shuffle(order)
        folds = [order[i::kk] for i in range(kk)]

        raw_ious, ref_ious = [], []
        thr_default = det.pix_thr if det.pix_thr is not None else 0.0
        for fi in range(kk):
            hold = folds[fi]
            tr = [i for i in range(n) if i not in set(hold)]
            net = self._train_one(det, [defect_imgs[i] for i in tr], [base_logits[i] for i in tr],
                                  [defect_masks[i] for i in tr], steps=self.steps, seed=self.seed + fi)
            for i in hold:
                bl = base_logits[i]
                gt = _mask_to(defect_masks[i], *bl.shape)
                raw_ious.append(_per_image_iou([(bl >= thr_default).astype(np.uint8)], [gt]))
                rl = self._apply(net, det, defect_imgs[i], bl)
                ref_ious.append(_per_image_iou([(rl >= thr_default).astype(np.uint8)], [gt]))
        if not raw_ious:
            return
        raw_m, ref_m = float(np.mean(raw_ious)), float(np.mean(ref_ious))
        self.gain = ref_m - raw_m
        if self.gain <= margin:
            return                                             # 净增益不够,保持关闭(零回退)
        # 启用:全fit数据训最终头
        self.net = self._train_one(det, defect_imgs, base_logits, defect_masks,
                                   steps=self.steps, seed=self.seed)
        self.enabled = True

    @torch.no_grad()
    def refine(self, det, img, base_logit_hw):
        """生产入口:未启用直接原样返回base_logit,零回退。"""
        if not self.enabled or self.net is None:
            return base_logit_hw
        return self._apply(self.net, det, img, base_logit_hw)
