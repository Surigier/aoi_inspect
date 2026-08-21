class ActiveLearningLoop:
    """主动学习闭环:维护正常/缺陷样本集,操作员反馈后重跑少样本适配
    (记忆库方法无需梯度训练,重建库 + 重标定阈值即完成在线更新)。

    defect_masks 可选:传入后与 defect_images 一一对应增长,反馈时若带掩膜就一并
    append,调用 adapter.fit_fewshot(normals, defects, defect_masks=masks)——这样
    同一个类兼容两种 adapter:旧的纯记忆库(fit_fewshot(normals, defects))和现在
    生产用的 CompetitionLargeDetector(fit_fewshot(normals, defects, defect_masks=...),
    训监督分割头/SAM/crop_cascade/component_graph等全部OOF门控子模块)。不传
    defect_masks 时行为与之前完全一致(向后兼容,原有测试不用改)。"""

    def __init__(self, adapter, normal_images, defect_images, defect_masks=None):
        self.adapter = adapter
        self.normals = list(normal_images)
        self.defects = list(defect_images)
        self.masks = list(defect_masks) if defect_masks is not None else None
        self._refit()

    def _refit(self, retrain_ead=True):
        """retrain_ead=False:跳过EAD学生重训(见下面feedback()的说明)。只有生产
        CompetitionLargeDetector支持这个参数,旧的纯记忆库adapter不认——所以只在
        真的要跳过时才传,默认路径的调用签名和以前一字不差(向后兼容)。"""
        kw = {}
        if self.masks is not None:
            kw["defect_masks"] = self.masks
        if not retrain_ead:
            kw["retrain_ead"] = False
        self.adapter.fit_fewshot(self.normals, self.defects, **kw)

    def predict(self, image):
        """image: (1,3,H,W) -> (BranchResult, is_defect)"""
        return self.adapter.predict(image)

    def feedback(self, image, is_defect, mask=None):
        """image: (3,H,W) 单图;is_defect=操作员判定的真实标签;mask 仅在
        is_defect=True 且本loop启用了掩膜追踪时使用(缺陷图掩膜,提升定位精度)。
        返回更新后的 (正常集大小, 缺陷集大小)。

        **实时性**:赛题原文要求"当系统**误检或漏检**时,操作员可提供**实时**反馈"——
        "实时"修饰的是两者,所以两条路都跳过EAD学生重训(retrain_ead=False),其余标定
        (阈值/分割头/DINO门/框/像素阈值)全部照常重跑。

        漏检(is_defect=True):只新增缺陷图,而EAD学生只在正常图上训、缺陷图仅参与
        阈值标定,跳过学生重训在原理上就是无损的。

        误检(is_defect=False):新增的是正常图,直觉上"学生该重训才能吃到新样本",
        但实测证伪了这个直觉(scripts/run_fp_margin.py,hazelnut):把留出正常图里
        融合分最高(最接近被误判)的3张标记为误检后,**快路径的安全边距改善
        +0.326、完整路径只有+0.269,快路径反而更好**。机制上说得通:修复误检靠的是
        阈值/DINO门/像素阈值重标,这些通路都不经过学生权重;而重训学生会让学生把这张
        新正常图学进去、重建得更好→它的EAD分下降→阈值标定时它不再是"高分正常样本",
        阈值反而少上移一点。所以误检也走快路径,既实时又不损效果。

        泛化性附注:同实验里未被反馈的其余正常图边距只改善+0.021——反馈主要修被标记
        的那张及其近邻,不会因为几张误检反馈就把整体灵敏度调没,这是期望行为。

        学生权重的陈旧性由离线完整重训兜底(fit_fewshot默认retrain_ead=True),不占用
        操作员的实时交互路径。"""
        if is_defect:
            self.defects.append(image)
            if self.masks is not None:
                import numpy as np
                self.masks.append(mask if mask is not None else np.zeros((8, 8), "uint8"))
        else:
            self.normals.append(image)
        self._refit(retrain_ead=False)             # 两条路都走实时快路径(见上docstring实测依据)
        return len(self.normals), len(self.defects)
