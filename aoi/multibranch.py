import torch
from .fewshot import FewShotAdapter
from .fusion import znorm, auroc


class MultiBranchAdapter:
    """多分支:留出法估各分支对未见正常图的可分性,硬选最可靠分支,再标定单一阈值。
    接口与 FewShotAdapter 一致,可直接喂给 run_protocol。"""

    def __init__(self, branches):
        self.branches = branches
        self.stats = []        # [(mean, std), ...] 每分支正常分统计
        self.weights = []      # 每分支可靠性权重(由现场缺陷估计)
        self.threshold = None

    def fit_fewshot(self, normal_images, defect_images):
        # 1) 留出法估可靠性:一部分正常图做验证,看哪个分支对"未见正常图"不误报。
        #    关键:在那 30 张明显缺陷上两分支都≈满分,区分不出泛化;真正差距在对未见正常的误报。
        k = max(1, len(normal_images) // 5)
        val_normals, build_normals = normal_images[:k], normal_images[k:]
        if len(build_normals) == 0:                      # 样本太少则退回全量
            build_normals, val_normals = normal_images, normal_images
        # 监督分支(判别头)须在**未见缺陷**上评可靠性(否则背题作弊),故缺陷拆 build/val;
        # 无监督分支不碰缺陷,用**全部缺陷**评更稳(否则验证缺陷太少→噪声大→总选纹理)。
        kd = max(1, len(defect_images) // 5)
        val_defects, build_defects = defect_images[:kd], defect_images[kd:]
        if len(build_defects) == 0:
            build_defects, val_defects = defect_images, defect_images
        self.weights = []
        for b in self.branches:
            supervised = hasattr(b, "fit_supervised")
            self._fit_branch(b, build_normals, build_defects)
            score_defects = val_defects if supervised else defect_images
            vn = [b.infer(x.unsqueeze(0)).score for x in val_normals]
            ds = [b.infer(d.unsqueeze(0)).score for d in score_defects]
            sep = auroc(vn + ds, [0] * len(vn) + [1] * len(ds))
            self.weights.append(max(0.0, sep - 0.5))
        if sum(self.weights) <= 1e-9:
            self.weights = [1.0] * len(self.branches)
        # 2) 用全部正常图做最终拟合 + 统计正常分均值方差
        self.stats = []
        for b in self.branches:
            self._fit_branch(b, normal_images, defect_images)
            ns = [b.infer(img.unsqueeze(0)).score for img in normal_images]
            m = sum(ns) / len(ns)
            var = sum((x - m) ** 2 for x in ns) / len(ns)
            self.stats.append((m, var ** 0.5))
        # 3) 在融合分上标定阈值
        norm_fused = [self._fused(img.unsqueeze(0))[0] for img in normal_images]
        def_fused = [self._fused(img.unsqueeze(0))[0] for img in defect_images]
        self.threshold = FewShotAdapter._calibrate(norm_fused, def_fused)
        return self.threshold

    @staticmethod
    def _fit_branch(b, normals_list, defects_list):
        """无监督分支用 fit(正常);监督分支(判别头)用 fit_supervised(正常, 缺陷)。"""
        if hasattr(b, "fit_supervised"):
            b.fit_supervised(normals_list, defects_list)
        else:
            b.fit(torch.stack(normals_list))

    def _fused(self, image):
        """按可分性硬选最可靠分支(各分支 z 幅度不可比,加权平均会被大幅度分支带偏;
        改为用现场 30 缺陷选出对该域最判别的分支)。返回 (该分支 z, 其 BranchResult)。"""
        idx = max(range(len(self.weights)), key=lambda i: self.weights[i])
        b, (m, s) = self.branches[idx], self.stats[idx]
        r = b.infer(image)
        return znorm(r.score, m, s), r

    def predict(self, image):
        fused, res = self._fused(image)
        is_defect = bool(fused >= self.threshold)
        res.score = fused            # 让 AUROC/阈值都基于融合分(保留最异常分支的 anomaly_map)
        res.defect_type = res.defect_type if is_defect else "normal"
        return res, is_defect
