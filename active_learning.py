class ActiveLearningLoop:
    """主动学习闭环:维护正常/缺陷样本集,操作员反馈后重跑少样本适配。

    **只服务一种适配器契约**——生产的 CompetitionLargeDetector:

        fit_fewshot(normals, defects, defect_masks=None, retrain_ead=True)

    早期版本为了同时兼容旧的纯记忆库适配器,在 _refit 里按参数有无拼 kwargs。那套
    分支在"两条反馈路都改走快路径"之后就失效了(每次都会传 retrain_ead,旧适配器
    直接 TypeError),而且它掩盖了一个事实:本闭环的实测结论(快路径耗时1193s→251s、
    误检反馈快路径边距+0.326优于完整路径+0.269)全部是在 CompetitionLargeDetector
    上测的,换个适配器这些结论一条都不成立。所以这里不做适配,只认一种契约——
    交付物里只留一条真实路径。

    defect_masks 可选:传入后与 defect_images 一一对应增长,反馈时若带掩膜就一并
    append,一路带进 fit_fewshot 去训监督分割头/SAM/crop_cascade 等 OOF 门控子模块。"""

    def __init__(self, adapter, normal_images, defect_images, defect_masks=None):
        self.adapter = adapter
        self.normals = list(normal_images)
        self.defects = list(defect_images)
        self.masks = list(defect_masks) if defect_masks is not None else None
        self._refit()

    def _refit(self, retrain_ead=True):
        """retrain_ead=False:跳过EAD学生重训(依据见下面 feedback() 的说明)。
        无条件按生产契约调用,不做任何签名探测。"""
        self.adapter.fit_fewshot(self.normals, self.defects,
                                 defect_masks=self.masks, retrain_ead=retrain_ead)

    def predict(self, image):
        """image: (1,3,H,W) -> CompetitionLargeDetector.predict() 的判决字典
        (score / is_defect / defect_type / _raws)。要像素图和检测框用 adapter.locate()。"""
        return self.adapter.predict(image)

    def feedback(self, image, is_defect, mask=None):
        """image: (3,H,W) 单图;is_defect=操作员判定的真实标签;mask 仅在
        is_defect=True 且本loop启用了掩膜追踪时使用(缺陷图掩膜,提升定位精度)。
        返回更新后的 (正常集大小, 缺陷集大小)。反馈缺陷且带掩膜时,VLM 的即时诊断
        (现象一句话 + 类别)会放在 `self.last_diagnosis`,供界面直接展示给操作员。

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
        diag = None
        if is_defect and mask is not None:
            # **反馈时的即时诊断**(冷路径,不计时):用VLM说出"这是什么缺陷",
            # 而不是只把样本塞进集合。赛题要求"操作员反馈→系统可回溯检测逻辑",
            # 而 explain() 回溯的是模型内部分数链路(给开发者看),操作员需要的是
            # 一句人话。VLM不可用时静默返回None,只记录样本,不影响任何已有行为。
            try:
                from .vlm_type import diagnose_defect
                ref = self.normals[0] if self.normals else None
                phen, typ = diagnose_defect(image, mask, normal_ref=ref)
                if phen:
                    diag = {"现象": phen, "类型": typ}
            except Exception:
                diag = None
        self.last_diagnosis = diag
        if is_defect:
            self.defects.append(image)
            if self.masks is not None:
                import numpy as np
                self.masks.append(mask if mask is not None else np.zeros((8, 8), "uint8"))
        else:
            self.normals.append(image)
        self._refit(retrain_ead=False)             # 两条路都走实时快路径(见上docstring实测依据)
        return len(self.normals), len(self.defects)
