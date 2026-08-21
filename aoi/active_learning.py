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

        **实时性**:赛题要求操作员能提供"实时反馈"。反馈**漏检**(is_defect=True)时
        只新增缺陷图,而EAD学生只在正常图上训、缺陷图仅参与阈值标定——所以这一路
        跳过学生重训(retrain_ead=False),把每次反馈从分钟级压到秒级,其余标定
        (阈值/分割头/DINO门/框/像素阈值)全部照常重跑,判定质量不打折。
        反馈**误检**(is_defect=False)时新增的是正常图,EAD学生必须重训才能吃到这个
        新正常样本,这一路走完整fit。"""
        if is_defect:
            self.defects.append(image)
            if self.masks is not None:
                import numpy as np
                self.masks.append(mask if mask is not None else np.zeros((8, 8), "uint8"))
        else:
            self.normals.append(image)
        self._refit(retrain_ead=not is_defect)     # 只新增缺陷图→学生无需重训
        return len(self.normals), len(self.defects)
