class ActiveLearningLoop:
    """主动学习闭环:维护正常/缺陷样本集,操作员反馈后重跑少样本适配
    (记忆库方法无需梯度训练,重建库 + 重标定阈值即完成在线更新)。"""

    def __init__(self, adapter, normal_images, defect_images):
        self.adapter = adapter
        self.normals = list(normal_images)
        self.defects = list(defect_images)
        self.adapter.fit_fewshot(self.normals, self.defects)

    def predict(self, image):
        """image: (1,3,H,W) -> (BranchResult, is_defect)"""
        return self.adapter.predict(image)

    def feedback(self, image, is_defect):
        """image: (3,H,W) 单图;is_defect=操作员判定的真实标签。
        返回更新后的 (正常集大小, 缺陷集大小)。"""
        (self.defects if is_defect else self.normals).append(image)
        self.adapter.fit_fewshot(self.normals, self.defects)
        return len(self.normals), len(self.defects)
