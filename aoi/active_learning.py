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

    def _refit(self):
        if self.masks is not None:
            self.adapter.fit_fewshot(self.normals, self.defects, defect_masks=self.masks)
        else:
            self.adapter.fit_fewshot(self.normals, self.defects)

    def predict(self, image):
        """image: (1,3,H,W) -> (BranchResult, is_defect)"""
        return self.adapter.predict(image)

    def feedback(self, image, is_defect, mask=None):
        """image: (3,H,W) 单图;is_defect=操作员判定的真实标签;mask 仅在
        is_defect=True 且本loop启用了掩膜追踪时使用(缺陷图掩膜,提升定位精度)。
        返回更新后的 (正常集大小, 缺陷集大小)。"""
        if is_defect:
            self.defects.append(image)
            if self.masks is not None:
                import numpy as np
                self.masks.append(mask if mask is not None else np.zeros((8, 8), "uint8"))
        else:
            self.normals.append(image)
        self._refit()
        return len(self.normals), len(self.defects)
