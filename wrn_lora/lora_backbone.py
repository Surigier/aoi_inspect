"""WRN定位骨干(aoi.backbone.Backbone layers(1,2))的Conv-LoRA适配:只在layer2最后
1~2个Bottleneck的conv2(3×3空间卷积)插旁路,layer1和layer2前部完全冻结,BN全程
eval()冻结(不更新running统计量——小批量会直接破坏预训练特征,这是要害)。

【隔离原则同rddn_yolo/】不改动aoi/下任何文件,只读复用aoi.backbone.Backbone构造
骨干后在这里做手术。"""
import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """冻结原conv + 低秩空间卷积旁路(down:Cin→r 1×1,up:r→Cout保持原kernel_size,
    这样LoRA修正的是空间卷积本身而不只是通道混合——用户明确要求"改空间卷积,不只是
    通道映射",对划痕/裂纹这类有方向性的缺陷更有意义)。up权重零初始化,起点≡冻结base。"""

    def __init__(self, base_conv: nn.Conv2d, r=4, alpha=1.0):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_ch, out_ch = base_conv.in_channels, base_conv.out_channels
        k = base_conv.kernel_size
        self.down = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.up = nn.Conv2d(r, out_ch, kernel_size=k, stride=base_conv.stride,
                           padding=base_conv.padding, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))

    @torch.no_grad()
    def merged_conv(self):
        """把LoRA旁路合并回一个普通nn.Conv2d(推理时不增加分支/延时的关键)。
        base(x)+scale*up(down(x)): down是1×1,up是k×k,合并后等价于一个k×k conv,
        权重=base.weight + scale*(up.weight与down.weight的等效卷积核)。"""
        base = self.base
        out_ch, in_ch, kh, kw = base.weight.shape
        # up.weight:(out_ch,r,kh,kw), down.weight:(r,in_ch,1,1) -> 等效(out_ch,in_ch,kh,kw)
        eq = torch.einsum("orhw,ri->oihw", self.up.weight, self.down.weight[:, :, 0, 0])
        merged = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), stride=base.stride,
                          padding=base.padding, bias=base.bias is not None)
        merged.weight.copy_(base.weight + self.scale * eq)
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        return merged


def build_lora_wrn(device="cuda", n_late_blocks=1, r=2, alpha=1.0):
    """返回(backbone对象,extractor函数,lora_modules列表)。n_late_blocks=1→只改
    layer2最后1块(layer2.3.conv2);2→layer2.2.conv2+layer2.3.conv2。r=2主实验/4挑战组。"""
    from aoi.backbone import Backbone
    bb = Backbone(layers=(1, 2), device=device)

    # 全部冻结+BN永久eval(不更新running统计量)
    for p in bb.model.parameters():
        p.requires_grad_(False)
    bb.model.eval()

    lora_modules = []
    block_idxs = [3] if n_late_blocks == 1 else [2, 3]
    for bi in block_idxs:
        block = bb.model.layer2[bi]
        wrapped = LoRAConv2d(block.conv2, r=r, alpha=alpha).to(device)
        block.conv2 = wrapped
        lora_modules.append(wrapped)

    lora_params = []
    for lm in lora_modules:
        lora_params += list(lm.down.parameters()) + list(lm.up.parameters())

    def extractor(img, seg_in=512):
        """等价于aoi.competition._wrn_feats,但不加@torch.no_grad()——LoRA参数需要梯度。
        bb.model整体仍在eval()模式(BN冻结),只有LoRA的down/up的requires_grad=True。"""
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = x.to(device)
        x = torch.nn.functional.interpolate(x, size=(seg_in, seg_in), mode="bilinear", align_corners=False)
        feats = bb.model(x)
        size = feats[0].shape[-2:]
        feats = [torch.nn.functional.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats]
        return torch.cat(feats, dim=1)[0]

    return bb, extractor, lora_modules, lora_params
