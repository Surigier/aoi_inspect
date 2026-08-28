"""VLM监督的缺陷类型归属头:**fit期**用VLM给30张缺陷图打类型标签,在位置匹配
特征上训一个最近质心分类器;**推理期零API依赖、零外网依赖**。

为什么这么拆(而不是推理期直接问VLM):赛题"测试集运行前...进行迁移学习"——fit
不计时,而locate()要在200ms内返回。一次API往返500ms~2s直接爆预算,且判分机器可能
无外网。所以把VLM的能力在fit期"蒸馏"进一个几十字节的质心表里。

特征用的是位置匹配的4维z分(见scripts/diag_type_locmatch.py验证):
  z_appear 掩膜内梯度幅值偏移   z_color 掩膜内色度偏移
  z_dim    整图前景面积全局z     z_struct 掩膜内WRN深层特征偏移
这4维单靠argmax能到58%;这里换成VLM标签监督的质心分类,让它学到"色彩变化=色度高
**且**梯度低"这种组合判据,而不是只看谁最大。

关键工程约束:零假设参考图的特征必须**在fit期预算好并缓存**。原始做法是推理时对
30张正常图各跑一次WRN(30×9ms=270ms,直接爆预算);且768×128²×30=1.5GB显存,2060
放不下。所以深层参考特征缓存在32²粗格(94MB)——掩膜是块状区域,32²的空间分辨率
足够做"在这块位置上池化",不需要128²。

降级路径:VLM不可用(无key/无网/超时)→ fit()返回False → competition.py自动退回
原有的_ztype启发式。**绝不能让整个检测崩掉。**
"""
import numpy as np
import torch
import torch.nn.functional as F

from .branches.color_ad import rgb_to_lab
from .fusion import znorm
from .vlm_type import TYPES, label_defect_types

SZ = 320        # 浅层特征(色度/梯度)工作分辨率
DEEP_G = 32     # 深层参考特征缓存网格
N_REF = 30      # 零假设参考正常图张数


def _down(img, size):
    x = img.unsqueeze(0) if img.dim() == 3 else img
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _shallow_maps(img):
    """(3,H,W)[0,1] → 色度图(2,SZ,SZ) 和 梯度幅值图(SZ,SZ),都在SZ²上算。"""
    x = _down(img, SZ).cpu()
    lab = rgb_to_lab(x)[0]                                    # (3,SZ,SZ)=L,a,b
    gray = x[0].mean(0, keepdim=True)[None]
    gx = F.conv2d(gray, torch.tensor([[[[-1., 0., 1.]]]]), padding=(0, 1))
    gy = F.conv2d(gray, torch.tensor([[[[-1.], [0.], [1.]]]]), padding=(1, 0))
    grad = (gx ** 2 + gy ** 2).sqrt()[0, 0]
    return lab[1:3].contiguous(), grad.contiguous()


MAX_MASK_PX = 2048          # 掩膜内参与池化的像素上限(见_mask_at)


def _mask_at(mask, sz, cap=None):
    """掩膜重采样到sz²布尔图;全空时退回全图(宁可信号弱,不要除零)。

    **cap:掩膜像素上限**。参考特征池化 `self._ref_deep[:, :, m]` 的开销与掩膜面积
    **成正比**——(30张,768通道,像素数)的中间张量,掩膜大时会炸。
    逐段计时实测:掩膜大的那轮类型头中位 **89.4ms**(占满管线35%,比图级判决还贵),
    掩膜紧的那轮只要 17.8ms。
    做法:超过上限就均匀抽样到上限。类型判别用的是掩膜内特征的**均值**,
    2048个像素估均值已经足够稳(标准误~1/√2048),多取纯属浪费。"""
    m = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                      size=(sz, sz), mode="area")[0, 0] > 0.02
    if not m.any():
        return torch.ones(sz, sz, dtype=torch.bool)
    if cap:
        idx = m.flatten().nonzero(as_tuple=True)[0]
        if len(idx) > cap:
            sel = idx[torch.linspace(0, len(idx) - 1, cap).long()]
            m = torch.zeros(sz * sz, dtype=torch.bool)
            m[sel] = True
            m = m.view(sz, sz)
    return m


def _z_vec(q, R):
    """统一口径:z =(测试图到正常均值的距离 - 正常图之间的典型距离)/ 正常距离的散布。
    **必须减掉零假设的典型距离**:正常图彼此本来就有非零距离,而深层特征768维、
    高维范数高度集中(典型距离大、散布极小),不减均值会让结构维系统性通吃。"""
    mu = R.mean(0)
    dn = (R - mu).norm(dim=1)
    return (float((q - mu).norm()) - float(dn.mean())) / (float(dn.std()) or 1.0)


class VLMTypeHead:
    """用法:fit(det, normals, defects, masks) → True表示可用;
    predict(det, img, mask, raws) → 5类之一。fit返回False时调用方须走_ztype。"""

    def __init__(self):
        self.ready = False
        self.rule_mode = False     # VLM不可用时转规则模式(位置匹配特征argmax,离线58%)
        self.centroids = None      # (K,4) 标准化空间里的类质心
        self.classes = None        # 长度K的类名
        self.fmu = self.fsd = None # 4维特征的标准化统计(否则量纲大的维通吃欧氏距离)
        self.labels = None         # VLM给fit缺陷图打的标签(留档,供explain/复盘)
        self._ref_chroma = self._ref_grad = self._ref_deep = None

    # ---------- fit ----------
    def _cache_refs(self, det, normals):
        ref = normals[:N_REF]
        self._ref_chroma, self._ref_grad, deep = [], [], []
        for x in ref:
            c, g = _shallow_maps(x)
            self._ref_chroma.append(c); self._ref_grad.append(g)
            f = det._wrn_feats(x)                              # (768,128,128)
            deep.append(F.interpolate(f[None], size=(DEEP_G, DEEP_G), mode="area")[0].cpu())
        self._ref_chroma = torch.stack(self._ref_chroma)       # (R,2,SZ,SZ)
        self._ref_grad = torch.stack(self._ref_grad)           # (R,SZ,SZ)
        self._ref_deep = torch.stack(deep)                     # (R,768,G,G)

    def _feat(self, det, img, mask, raws=None):
        """4维位置匹配特征。参考图特征全部走fit期缓存,推理只多一次浅层计算。"""
        m_s = _mask_at(mask, SZ, MAX_MASK_PX)
        m_d = _mask_at(mask, DEEP_G)          # 32²=1024像素,本来就在上限内
        chroma, grad = _shallow_maps(img)
        q_c = chroma[:, m_s].mean(1)
        q_g = float(grad[m_s].mean())
        f = det._wrn_feats(img)
        q_d = F.interpolate(f[None], size=(DEEP_G, DEEP_G), mode="area")[0].cpu()[:, m_d].mean(1)
        r_c = self._ref_chroma[:, :, m_s].mean(2)              # (R,2)
        r_g = self._ref_grad[:, m_s].mean(1).numpy()           # (R,)
        r_d = self._ref_deep[:, :, m_d].mean(2)                # (R,768)
        z_appear = (q_g - float(r_g.mean())) / (float(r_g.std()) or 1.0)
        z_color = _z_vec(q_c, r_c)
        z_struct = _z_vec(q_d, r_d)
        raw_dim = raws[2] if raws is not None else det.branches[2].score(img)
        z_dim = znorm(raw_dim, *det.stats[2])
        return np.array([z_appear, z_color, z_dim, z_struct], dtype=np.float64)

    def fit(self, det, normals, defects, defect_masks, verbose=True):
        if not defects or defect_masks is None:
            return False
        labels = label_defect_types(defects, defect_masks, verbose=verbose,
                                    normal_ref=normals[0] if normals else None)
        self._cache_refs(det, normals)
        if labels is None:
            # **离线降级:规则模式**。VLM不可用(无外网/无key/超时)时,不要退回
            # competition._ztype那套"检测分支z分argmax"——那套只有38%,因为检测分数是为
            # **检测**设计的(EAD对任何缺陷都强响应),不携带类型信息。
            # 改用本模块已有的**位置匹配4维特征**直接argmax:同样零网络依赖,实测58%。
            # 赛委机器无外网是工业评测的常态,这条降级路径的质量直接决定赛场表现。
            self.rule_mode = True
            self.ready = True
            if verbose:
                print("!! VLM不可用 → 类型头转入**规则模式**(位置匹配特征argmax,离线可用)",
                      flush=True)
            return True
        self.rule_mode = False
        X, y = [], []
        for img, mk, lab in zip(defects, defect_masks, labels):
            if lab is None:                                    # 单张标注失败就跳过这张,不拖垮整体
                continue
            X.append(self._feat(det, img, mk)); y.append(lab)
        if not y:
            return False
        # 注意:标签只覆盖1类**不是失败,是信息**。赛题给的30张缺陷图就是该产品缺陷的
        # 代表样本,全是同一类说明这个产品的缺陷就是这一类,此时最近质心自然退化成
        # "恒定输出该类"——那正是最优估计。实测hazelnut的17张fit缺陷图VLM全标成
        # "常见外观缺陷",按"少于2类就弃用"处理会退回启发式只拿8/11,而恒定输出是11/11。
        # 单产品迁移场景下缺陷类型本来就集中,这是主流情况不是边角料。
        X = np.stack(X)
        self.fmu = X.mean(0); self.fsd = X.std(0) + 1e-9
        Xn = (X - self.fmu) / self.fsd
        self.classes = sorted(set(y), key=lambda t: TYPES.index(t) if t in TYPES else 99)
        self.centroids = np.stack([Xn[[i for i, t in enumerate(y) if t == c]].mean(0)
                                   for c in self.classes])
        self.labels = labels
        self.ready = True
        if verbose:
            from collections import Counter
            print(f"!! VLM类型头就绪:{len(y)}张样本 / {len(self.classes)}类 {dict(Counter(y))}", flush=True)
        return True

    # ---------- 推理 ----------
    # 规则模式下4维特征各自对应的赛题类型(顺序与_feat返回的一致)
    RULE_NAMES = ["常见外观缺陷", "色彩变化", "尺寸偏差", "缺件少件"]

    def predict(self, det, img, mask, raws=None):
        if not self.ready:
            return None
        try:
            f = self._feat(det, img, mask, raws)
            rule = self.RULE_NAMES[int(np.argmax(f))]
            if getattr(self, "rule_mode", False):
                return rule                                    # 离线规则模式
            # 【已判负,勿再加】曾试"rule指向质心表达不了的类型时采信rule"来救那三个0%的类:
            # 5类目混合实测 常见外观缺陷 67.6%→19.6%、合计 48.2%→19.4%,
            # 而三个0%的类**一个都没救回来**。原因:规则特征的argmax经常指向"尺寸偏差"
            # (该类不在质心类别里)→ 触发兜底 → 把本该正确的"常见外观缺陷"覆盖掉;
            # 而真正的尺寸偏差图,规则特征又抓不到。**净负29个百分点。**
            z = (f - self.fmu) / self.fsd
            d = ((self.centroids - z) ** 2).sum(1)
            return self.classes[int(np.argmin(d))]
        except Exception:
            return None                                        # 任何意外都不能让locate崩
