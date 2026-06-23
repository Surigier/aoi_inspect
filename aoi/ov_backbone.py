"""OpenVINO CPU 骨干:把 timm 特征提取器(含多层插值拼接)整体导出为 OV IR,
在 CPU 上推理,接口与 aoi.backbone.Backbone 一致(.extract(x)->(B,C,h,w))。
用于赛题"CPU<2s"挑战目标。动态 batch,供分块逐批前向。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ExtractModule(nn.Module):
    """复刻 Backbone.extract:多层特征上采样到同尺寸后按通道拼接。"""
    def __init__(self, timm_model):
        super().__init__()
        self.m = timm_model

    def forward(self, x):
        feats = self.m(x)
        size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                 for f in feats]
        return torch.cat(feats, dim=1)


class OVBackbone:
    def __init__(self, name: str = "resnet18", layers=(2, 3),
                 tile: int = 512, threads: int = 0):
        import timm
        import openvino as ov
        m = (timm.create_model(name, pretrained=True, features_only=True, out_indices=layers)
             .eval())
        wrap = _ExtractModule(m)
        with torch.no_grad():
            ov_model = ov.convert_model(wrap, example_input=torch.zeros(1, 3, tile, tile))
        ov_model.reshape([-1, 3, tile, tile])               # 动态 batch
        cfg = {"INFERENCE_NUM_THREADS": threads} if threads else {}
        self.compiled = ov.Core().compile_model(ov_model, "CPU", cfg)
        self.device = "cpu"

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.compiled(x.cpu().numpy())[0]
        return torch.from_numpy(out)
