import torch
import torch.nn.functional as F
import timm


class Backbone:
    """timm 多层特征提取器,把若干中间层上采样到同尺寸后按通道拼接。"""

    def __init__(self, name: str = "wide_resnet50_2", layers=(2, 3),
                 pretrained: bool = True, device: str = "cpu"):
        self.device = device
        self.model = (
            timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=layers)
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] -> (B, C_concat, h, w)"""
        x = x.to(self.device)
        feats = self.model(x)
        size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats]
        return torch.cat(feats, dim=1)
