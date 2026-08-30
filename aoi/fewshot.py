from typing import List, Tuple
import torch
from .types import BranchResult


GAP_RATIO = 10.0     # 空隙超过这个倍数才认为是病态,才下移阈值


class FewShotAdapter:
    """实现官方协议入口:用 100 正常 + 30 缺陷做迁移(建库 + 标定阈值)。"""

    def __init__(self, branch):
        self.branch = branch
        self.threshold = None

    def fit_fewshot(self, normal_images: List[torch.Tensor],
                    defect_images: List[torch.Tensor]) -> float:
        self.branch.fit(torch.stack(normal_images))
        normal_scores = [self.branch.infer(img.unsqueeze(0)).score for img in normal_images]
        defect_scores = [self.branch.infer(img.unsqueeze(0)).score for img in defect_images]
        self.threshold = self._calibrate(normal_scores, defect_scores)
        return self.threshold

    @staticmethod
    def _calibrate(normal_scores: List[float], defect_scores: List[float]) -> float:
        """选**平衡准确率**(TPR+TNR)/2 最高的阈值,再**把阈值从缺陷侧挪回空隙中间**。

        为什么要挪(实测,不是理论洁癖):候选阈值只来自**观测到的分数**。当两类分数
        之间存在大空隙时,空隙里一个候选都没有,阈值只能贴到缺陷侧的端点——也就是
        **30张fit缺陷图里分数最低的那张**。于是测试集里任何比"这30张中最弱的一张"
        还弱的缺陷,全部漏掉。

        真手机屏实测(scripts/diag_phone_detect.py):正常分0~3.35、缺陷分1.7亿~6.3亿,
        **两类零重叠**,阈值取3.35就能零误报100%召回;而标定出来的阈值是4.23亿
        (=fit缺陷最低分),直接切在缺陷分布中间 → **召回只有46%**。缺陷越小漏得越狠:
        最小25%那档召回仅4%,最大25%那档84%。

        错在**把阈值锚定在30张缺陷上,而不是100张正常图上**。正常图是有代表性的
        (赛题就给100张),30张缺陷只是缺陷总体的小样本、必然不含最弱的那些——
        锚在缺陷侧等于系统性保证漏检。

        挪的做法:找到best_t后,取"低于best_t的最高正常分"n_below,在(n_below, best_t)
        这段空隙里取中点。**在fit数据上可证明不会更差**:t只要仍大于n_below,TNR完全
        不变;t下移只会让更多缺陷越线,TPR不降反升。分数跨数量级时用几何中点,
        含0或负数时退回算术中点。空隙不存在(两类重叠)时n_below就紧贴best_t,
        中点≈best_t,**对分布重叠的类目零影响**。"""
        # ── CALIB_Q:把阈值锚定在**正常图分位数**上(默认关,>0 启用)──────────────
        # 为什么:原标定在"100正常+30缺陷"上找平衡准确率最优点,阈值必然落在某个观测
        # 分数上,而30张缺陷是小样本——抽到弱的阈值就低(误报涨)、抽到强的就高(漏检涨)。
        # 三种子实测:图级acc 在 0.669~0.913 摆动、误报率 7.7%~30.9%(差4倍),
        # **方差直接来自那30张的抽样**。而赛场上我们不控制评委给哪30张。
        # 分位数标定只依赖100张正常图(样本多、有代表性),误报率被 (1-q) 直接控住,
        # 与缺陷抽样解耦。保底:若该阈值连一半fit缺陷都拦不住,说明分位取得过高,
        # 下调到fit缺陷的中位数,避免走到"永不报警"的极端。
        import os as _os
        _q = float(_os.environ.get("CALIB_Q", "0"))
        if _q > 0 and len(normal_scores) >= 20:
            import numpy as _np
            t = float(_np.quantile(_np.asarray(normal_scores, float), _q))
            if defect_scores and sum(s >= t for s in defect_scores) < len(defect_scores) * 0.5:
                t = min(t, float(_np.median(_np.asarray(defect_scores, float))))
            return t

        candidates = sorted(set(normal_scores + defect_scores))
        n_pos = len(defect_scores)
        n_neg = len(normal_scores)
        # CALIB=plain:按**普通准确率**标定(赛题评的就是这个),自动吃到fit集的先验。
        # 为什么要改:平衡准确率把缺陷与正常等权,而真实测试流里正常图占绝大多数——
        # 三份独立实验显示同一个平衡准确率准则朝**两个相反方向**出错:
        #   2500²赛场级(100正常:30缺陷) 阈值偏低 → 误报80.2%,调对阈值 +0.272 acc
        #   混类原生单图(同上)          阈值偏低 → 误报25.7%,调对阈值 +0.108 acc
        #   真手机屏(20正常:30缺陷)     阈值偏高 → 漏检54%
        # 普通准确率天然按fit集的正常/缺陷比例加权:正常多则抬阈值少误报,
        # 缺陷多则压阈值少漏检——一个准则同时对症两个方向。
        import os as _os
        plain = _os.environ.get("CALIB", "bal") == "plain"
        best_t, best_bal = candidates[0], -1.0
        for t in candidates:
            tp = sum(s >= t for s in defect_scores)
            tn = sum(s < t for s in normal_scores)
            bal = ((tp + tn) / (n_pos + n_neg)) if plain else \
                  ((tp / n_pos) + (tn / n_neg)) / 2.0
            if bal >= best_bal:
                best_bal, best_t = bal, t
        import os as _os
        if _os.environ.get("CALIB_DEBUG"):
            _n = sorted(normal_scores); _d = sorted(defect_scores)
            _bl = [x for x in normal_scores if x < best_t]
            print(f"[calib] best_bal={best_bal:.4f} best_t={best_t:.6g} | "
                  f"正常 min={_n[0]:.6g} max={_n[-1]:.6g} | 缺陷 min={_d[0]:.6g} max={_d[-1]:.6g} | "
                  f"n_below={max(_bl) if _bl else None} | "
                  f"重叠={sum(1 for x in _d if x <= _n[-1])}/{len(_d)}", flush=True)
        below = [s for s in normal_scores if s < best_t]
        if not below:
            return best_t
        n_below = max(below)
        if n_below >= best_t:
            return best_t
        # 只在空隙"大到病态"时才下移。无条件下移会普遍抬高误报:实测第一版让手机屏
        # 召回46%→100%,但phone_battery图级acc 0.912→0.800。倍数门槛能干净分开两种
        # 情况——手机屏的空隙是1.26亿倍,正常类目通常只有2~3倍(不触发,行为与原来一致)。
        if n_below <= 0 or best_t / n_below < GAP_RATIO:
            return best_t
        return float((n_below * best_t) ** 0.5)              # 跨数量级时几何中点更居中

    def predict(self, image: torch.Tensor) -> Tuple[BranchResult, bool]:
        r = self.branch.infer(image)
        is_defect = bool(r.score >= self.threshold)
        r.defect_type = self.branch.defect_type if is_defect else "normal"
        return r, is_defect
