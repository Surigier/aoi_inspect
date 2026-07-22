"""组件图逻辑异常分支(UniVAD CVPR2025组件建模思想的轻量落地):
纹理seg-head的死穴是"局部都正常、整体错了"的逻辑缺陷(缺件/错序/错位)——逐像素
特征在缺件位置看到的是正常背景。对症:组件级建模。
- fit期(不计时):SAM everything模式在模板正常图上生成组件伪标签(重模型只在这里),
  ECC对齐其余正常图,在每个组件ROI池化现有WRN特征,建每组件正常分布(mu+距离统计)。
- 推理期(热路径):ECC对齐(~几ms)+ 复用locate()已算的WRN特征图做ROI池化(近零开销),
  逐组件z-score超线→并入该组件模板掩膜(warp到测试帧)作为定位。
- 曾试Hungarian全局匹配(use_hungarian=True,槽位↔身份最优对应,理论上能区分"缺件"
  vs"错序互换"两种异常),2026-07-20同一fit受控A/B(run_comp_graph_v1v2.py)判负:
  LOCO 5类均值Δ含漏检 z-score版+0.016 vs Hungarian版+0.004,理论动机未在真实数据
  兑现(单一全局刚性ECC对齐限制了错序检测空间,"并集贴两块"策略判断不准时反而多贴
  伤精度)。默认关,代码留作opt-in研究件,见__init__注释。
- 门控:fit留出(30张逻辑缺陷掩膜)验证并集掩膜IoU净增益>0.01才启用,否则zero-footprint
  ——同SAM门/crop_cascade一致的"opt-in,OOF验证,零回退"哲学。"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _ecc_warp(gray_test, gray_tmpl, size=256):
    """test→template 的欧氏warp(2x3),失败返回None。输入都是原尺度灰度uint8。"""
    try:
        g1 = cv2.resize(gray_test, (size, size))
        g2 = cv2.resize(gray_tmpl, (size, size))
        warp = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(g1, g2, warp, cv2.MOTION_EUCLIDEAN,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
        return warp
    except Exception:
        return None


def _gray_u8(img):
    return (img.mean(0).cpu().numpy() * 255).astype(np.uint8)


class ComponentGraph:
    """fit见模块docstring。所有掩膜/特征在128²WRN特征格上操作(与seg_head同格)。"""

    def __init__(self, device="cuda", max_comps=12, z_thr=3.0, use_hungarian=False):
        self.device = device
        self.max_comps = max_comps
        self.z_thr = z_thr
        # use_hungarian默认False(2026-07-20受控A/B判负,见下)。
        # 曾试Hungarian(v2,run_comp_graph_v1v2.py同一fit隔离算法差异做受控对比):LOCO 5类
        # 均值Δ含漏检 v1=+0.016 vs v2=+0.004,v2在4/5类持平或更差,唯一"赢"的pushpins框命中
        # (+0.110 vs+0.084)伴随同类IoU更差(+0.034 vs+0.068,此消彼长非净赢)。理论动机
        # (抓组件互换/错序)未在真实数据兑现——单一全局刚性ECC对齐本身限制了错序检测空间
        # (真正错位需局部搜索,未实现),"并集贴两块"策略在判断不够准时多贴不该贴的像素反而
        # 伤精度。Hungarian代码留作opt-in研究件,不默认开。
        self.use_hungarian = use_hungarian
        self.enabled = False
        self.comp_masks = None                  # (K,gh,gw) bool,模板帧组件掩膜(特征格)
        self.tmpl_gray = None                   # 模板灰度(ECC用)
        self.mu = None                          # (K,C) 每组件正常特征均值
        self.d_mu = None; self.d_sd = None      # (K,) 正常距离分布
        self.gain = None

    def _sam_components(self, tmpl_img):
        """SAM everything模式提模板组件掩膜(fit期一次,重模型不进热路径)。"""
        from .sam_refine import SamRefiner
        m = SamRefiner()._model()
        if m is None:
            return None
        arr = (tmpl_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        try:
            r = m.predict(arr, verbose=False, imgsz=1024)[0]     # 无提示=everything
        except Exception:
            return None
        if r.masks is None or len(r.masks.data) == 0:
            return None
        H, W = arr.shape[:2]
        masks = r.masks.data.cpu().numpy() > 0.5                 # (N,h,w)
        out = []
        area_img = masks.shape[1] * masks.shape[2]
        for mk in masks:
            a = mk.sum() / area_img
            if not (0.004 <= a <= 0.25):                         # 滤背景大块与噪点
                continue
            if any((mk & o).sum() / min(mk.sum(), o.sum()) > 0.6 for o in out):
                continue                                         # 去重(高重叠保先到=大者先)
            out.append(mk)
            if len(out) >= self.max_comps:
                break
        return out if len(out) >= 2 else None                    # <2个组件没有"图"可言

    @torch.no_grad()
    def _pool(self, feat, comp_masks_g):
        """feat(C,gh,gw) × (K,gh,gw)bool → (K,C) 掩膜均值池化。"""
        C = feat.shape[0]
        out = torch.zeros(len(comp_masks_g), C, device=feat.device)
        for i, mk in enumerate(comp_masks_g):
            idx = torch.from_numpy(mk).to(feat.device)
            if idx.sum() == 0:
                continue
            out[i] = feat[:, idx].mean(dim=1)
        return out

    def _warp_masks_g(self, warp_inv, gh, gw):
        """模板帧组件掩膜 → 测试帧(特征格)。warp是test→tmpl,贴回用WARP_INVERSE。"""
        if warp_inv is None:
            return self.comp_masks
        out = []
        for mk in self.comp_masks:
            w = cv2.warpAffine(mk.astype(np.uint8), warp_inv, (gw, gh),
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
            out.append(w > 0)
        return out

    def fit(self, det, normals, defect_imgs=None, defect_masks=None):
        """normals:正常图列表(≥16)。defect_*(可选):留出OOF验证门控用。"""
        self.enabled = False
        if not _HAS_CV2 or len(normals) < 16:
            return
        tmpl = normals[0]
        comps = self._sam_components(tmpl)
        if comps is None:
            return
        feat0 = det._wrn_feats(tmpl)
        gh, gw = feat0.shape[-2:]
        comp_g = [cv2.resize(mk.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
                  for mk in comps]
        comp_g = [mk for mk in comp_g if mk.sum() >= 4]
        if len(comp_g) < 2:
            return
        self.comp_masks = comp_g
        self.tmpl_gray = _gray_u8(tmpl)

        # 每组件正常特征分布:一半normals建mu,另一半标定距离分布(不混用,防自吹)
        half = min(20, len(normals) // 2)
        build, calib = normals[1:1 + half], normals[1 + half:1 + 2 * half]
        vecs = []                                            # list of (K,C)
        for n in build:
            warp = _ecc_warp(_gray_u8(n), self.tmpl_gray)
            f = det._wrn_feats(n)
            masks_n = self._warp_masks_g(warp, gh, gw)
            vecs.append(self._pool(f, masks_n))
        V = torch.stack(vecs)                                # (N,K,C)
        self.mu = V.mean(dim=0)                              # (K,C)
        dists = []
        for n in calib:
            warp = _ecc_warp(_gray_u8(n), self.tmpl_gray)
            f = det._wrn_feats(n)
            masks_n = self._warp_masks_g(warp, gh, gw)
            p = self._pool(f, masks_n)
            dists.append(torch.norm(p - self.mu, dim=1))     # (K,)
        D = torch.stack(dists)                               # (N,K)
        self.d_mu = D.mean(dim=0)
        self.d_sd = D.std(dim=0) + 1e-6

        if defect_imgs is None or defect_masks is None or len(defect_imgs) < 8:
            return                                           # 无标注不启用(没法验证净增益)
        # 门控设计定稿(完整证据链,防后人重蹈):
        # ①holdout用全量30张fit缺陷(组件统计只来自正常图→零泄漏;首版每3取1白白放大3倍方差)。
        # ②base用生产seg图(det.segment)。曾试OOF无偏base(seg_head.oof_maps)治"fit图是seg
        #   训练图→base过拟合好→低估边际增益"的偏差:juice_bottle估值-0.112→-0.037有改善,
        #   但仍测不出test真值+0.081,且把breakfast_box错判成+0.059(test真值-0.012)——fit侧
        #   对±0.1量级小信号两版都是掷硬币,此路确认到头。
        # ③生产base的保守偏差是特性不是bug:真·逻辑缺陷(局部纹理与正常相同)seg_head连fit图
        #   都记不住→base在fit侧也烂→门控才转正,恰好对应组件图该开的极端场景;而灾难级伤害
        #   (纹理类伪组件,sheet_metal强制开Δ=-0.218)门控估-0.169方向量级都对,能可靠拦截。
        #   代价:中间态(juice_bottle:局部可分但泛化差,test真值+0.081)会被错过——换取严格
        #   零回退,按项目纪律取安全侧。
        from .seg_head import _mask_to, _per_image_iou
        gains = []
        for i in range(len(defect_imgs)):
            img, mk = defect_imgs[i], defect_masks[i]
            amap = det.segment(img)
            thr = det.pix_thr if det.pix_thr is not None else float(amap.mean() + 3 * amap.std())
            fh, fw = amap.shape
            base = (amap >= thr).astype(np.uint8)
            gt = _mask_to(mk, fh, fw)
            merged = self.refine(det, img, base.copy(), feat=None)
            gains.append(_per_image_iou([merged], [gt]) - _per_image_iou([base], [gt]))
        self.gain = float(np.mean(gains)) if gains else -1.0
        self.enabled = self.gain > 0.01

    @torch.no_grad()
    def refine(self, det, img, mask, feat=None):
        """测试图:Hungarian找槽位↔身份全局最优对应(v2),按两种异常类型并入掩膜。
        feat可传locate()已算的WRN特征省一次前向;mask原样修改副本返回。"""
        if self.comp_masks is None or not _HAS_CV2:
            return mask
        f = feat if feat is not None else det._wrn_feats(img if img.dim() == 3 else img[0])
        gh, gw = f.shape[-2:]
        warp = _ecc_warp(_gray_u8(img if img.dim() == 3 else img[0]), self.tmpl_gray)
        masks_t = self._warp_masks_g(warp, gh, gw)
        p = self._pool(f, masks_t)                           # (K,C) 每槽位实际特征
        H, W = mask.shape
        out = mask

        def _paste(idx):
            mk = masks_t[idx].astype(np.uint8)
            rs = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            return rs

        if not self.use_hungarian:
            # v1(诊断/A-B对比用):槽位i只查自己身份i的正常范围,测不出错序互换。
            z = (torch.norm(p - self.mu, dim=1) - self.d_mu) / self.d_sd
            for i, zi in enumerate(z.tolist()):
                if zi >= self.z_thr:
                    out |= _paste(i)
            return out

        # v2默认:(K,K)代价矩阵,cost[i,j]=槽位i特征相对"身份j正常分布"的z分(不再只看i==j)
        dist = torch.cdist(p, self.mu)                       # (K,K) ||p_i - mu_j||
        cost = ((dist - self.d_mu[None, :]) / self.d_sd[None, :]).cpu().numpy()
        row, col = linear_sum_assignment(cost)               # row恒为0..K-1(方阵),col=π
        for i, j in zip(row, col):
            zij = float(cost[i, j])
            if zij >= self.z_thr:
                out |= _paste(i)                             # ①缺件/替换:配到最优身份仍不像→槽位i本身
            elif j != i:
                out |= _paste(i)                              # ②错序:槽位i长得像身份j≠i
                out |= _paste(j)                              #   ∪ 身份j的正常槽位(已空出的原位)
        return out
