"""Conv-LoRA:给YOLO检测头(model.model[-1],Detect模块)的cv2(框回归)/cv3(分类)
分支插入低秩旁路适配器,骨干+颈部(承载跨类预训练学到的"缺陷残差 vs 噪声"通用先验)
保持完全冻结。用于fit阶段(30张真缺陷+100张正常图配对负样本)的轻量适配——
可训练参数量远小于全量微调,是这个小样本场景下比"重训一整个crop-head"风险更低的
选择(crop_cascade同量级数据全量训练已经判负过一次,-0.059)。

旁路初始化up权重=0,起点上 base_conv(x)+0=base_conv(x),即刚开始等价于纯冻结模型,
未经真实fit前对生产零影响——同今天所有新机制一致的"零初始化残差/零回退"哲学。"""
import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """冻结原conv + 低秩1x1旁路(down:Cin→r,up:r→Cout)。1x1核足够表达通道间低秩修正,
    不需要对空间核也做低秩分解(降低参数量和实现复杂度)。"""

    def __init__(self, base_conv: nn.Conv2d, r=4, alpha=1.0):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_ch, out_ch = base_conv.in_channels, base_conv.out_channels
        self.down = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.up = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)              # 零初始化:起点≡冻结base,不会变差
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


def _wrap_conv_inplace(container, idx, r, alpha, lora_modules):
    """container[idx]可能是ultralytics.nn.modules.conv.Conv(有.conv子模块)或裸nn.Conv2d,
    两种情况都处理,原地替换成LoRA包装版。"""
    layer = container[idx]
    if isinstance(layer, nn.Conv2d):
        wrapped = LoRAConv2d(layer, r=r, alpha=alpha)
        container[idx] = wrapped
        lora_modules.append(wrapped)
    elif hasattr(layer, "conv") and isinstance(layer.conv, nn.Conv2d):
        layer.conv = LoRAConv2d(layer.conv, r=r, alpha=alpha)
        lora_modules.append(layer.conv)


def inject_lora_into_head(model, r=4, alpha=1.0):
    """detect.cv2(框回归)/cv3(分类)每个尺度的Sequential里所有Conv层原地替换成LoRA版。
    返回LoRA参数列表(供优化器只训这些)。"""
    detect = model.model[-1]
    lora_modules = []
    for branch_name in ("cv2", "cv3"):
        branch = getattr(detect, branch_name)
        for seq in branch:
            for i in range(len(seq)):
                _wrap_conv_inplace(seq, i, r, alpha, lora_modules)
    params = []
    for lm in lora_modules:
        params += list(lm.down.parameters()) + list(lm.up.parameters())
    return params, lora_modules


def freeze_all_except_lora(model):
    """骨干+颈部+检测头非LoRA部分全冻结,只留LoRA的down/up可训练。"""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            for p in m.down.parameters():
                p.requires_grad_(True)
            for p in m.up.parameters():
                p.requires_grad_(True)
