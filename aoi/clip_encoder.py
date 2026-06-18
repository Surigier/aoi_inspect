import torch
import torch.nn.functional as F
import open_clip

# OpenAI CLIP 预处理均值/方差
_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPEncoder:
    """open_clip 薄封装:输出 L2 归一化的图像/文本嵌入(CPU)。"""

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", device: str = "cpu"):
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.eval().to(device)
        self.device = device
        self._mean = torch.tensor(_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_STD).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode_text(self, prompts):
        tok = self.tokenizer(prompts).to(self.device)
        emb = self.model.encode_text(tok)
        return F.normalize(emb, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        x = F.interpolate(image.to(self.device), size=224, mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        emb = self.model.encode_image(x)
        return F.normalize(emb, dim=-1).cpu()
