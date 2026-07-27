"""FocalLoss+DiceLoss(二值分割版,干净重写,不是抄MultiADS的AGPL代码——FocalLoss是
2017年RetinaNet论文的标准公式,DiceLoss是医学分割里几十年的标准做法,两者都是通用
技术不是MultiADS的原创,重新推公式实现不涉及版权问题)。

动机:现有SupervisedSegHead._train_one()用BCEWithLogitsLoss(pos_weight=...)处理
"正样本(缺陷像素)极少"的类别不平衡,FocalLoss专门为这种极端不平衡设计(把权重集中
到难分类样本上),理论上比简单pos_weight缩放更适合pcb/phone_battery这种微小缺陷
(正样本像素少到个位数)。Dice直接优化重叠度,和IoU这个最终评分指标更贴近。

【已验证,判负】5类真实数据(pcb/phone_battery/cable/breakfast_box逻辑/pushpins
逻辑,见eval_focal_dice.py)ΔIoU=[-0.039,-0.016,+0.009,+0.032,-0.024],
median=-0.016 mean=-0.0076 min=-0.039,2/5类为正——不过关,中位数和均值都是负的。
**反直觉的地方**:理论上最该受益的微小缺陷类目(pcb/phone_battery)跌得最狠,
反而在breakfast_box(逻辑异常,不是微小缺陷)上涨最多——说明"FocalLoss该救微小
缺陷"这个理论预期在我们这套系统里没有兑现,具体原因未深究(可能是WRN特征本身
的表达能力才是瓶颈,损失函数怎么调都够不着;也可能是FocalDice的alpha/gamma超参
没针对我们的数据分布调过,直接用MultiADS默认值0.25/2.0)。默认不接入
competition.py,代码留opt-in研究件。
"""
import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t),标准二值focal loss。
    输入logit(未过sigmoid),target∈{0,1}同形状。"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logit, target):
        p = torch.sigmoid(logit)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        bce = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    """Dice = 2|X∩Y| / (|X|+|Y|),Loss = 1-Dice。输入logit,内部过sigmoid。"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logit, target):
        p = torch.sigmoid(logit)
        p_flat = p.reshape(p.shape[0], -1)
        t_flat = target.reshape(target.shape[0], -1)
        inter = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalDiceLoss(nn.Module):
    """MultiADS训练方式:(FocalLoss + DiceLoss)/2,各占一半。"""
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal = BinaryFocalLoss(alpha, gamma)
        self.dice = BinaryDiceLoss(smooth)

    def forward(self, logit, target):
        return (self.focal(logit, target) + self.dice(logit, target)) / 2.0
