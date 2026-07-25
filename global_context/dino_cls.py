"""DINOv2 CLS token提取(供EmbedAE用)。预处理与aoi/dino_gate.py的DinoGate完全一致
(518²/ImageNet归一化),这里额外取token 0(CLS)而不是patch tokens——生产真要接入
时这行代码可以直接摆进DinoGate._patches()同一次forward里顺手取,不需要额外前向;
这里为保持验证脚本独立(不改aoi/)单独建一份模型实例,增量延时的验证留到真要转正
那一步再测。"""
import torch
import torch.nn.functional as F
import timm

DINO_SZ = 518


class DinoCLS:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def cls(self, img):
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = x.to(self.device)
        x = F.interpolate(x, size=(DINO_SZ, DINO_SZ), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)
        return t[0, 0, :].float().cpu()                    # token 0 = CLS
