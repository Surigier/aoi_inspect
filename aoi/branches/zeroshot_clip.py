import time
import torch
import torch.nn.functional as F
from ..types import BranchResult

DEFAULT_NORMAL = [
    "a photo of a normal {}",
    "a photo of a flawless {}",
    "a photo of a {} without defect",
]
DEFAULT_ABNORMAL = [
    "a photo of a {} with defect",
    "a photo of a damaged {}",
    "a photo of a {} with anomaly",
]


class ZeroShotCLIPBranch:
    """零样本:用 CLIP 正常/异常文本提示与图像嵌入的相似度,softmax 得异常概率。"""
    defect_type = "appearance"

    def __init__(self, encoder, class_name: str = "object",
                 normal_prompts=None, abnormal_prompts=None, temperature: float = 0.01):
        self.encoder = encoder
        normal = [p.format(class_name) for p in (normal_prompts or DEFAULT_NORMAL)]
        abnormal = [p.format(class_name) for p in (abnormal_prompts or DEFAULT_ABNORMAL)]
        self._normal_emb = encoder.encode_text(normal)        # (Tn, D)
        self._abnormal_emb = encoder.encode_text(abnormal)    # (Ta, D)
        self.temperature = temperature

    def fit(self, images: torch.Tensor):
        """零样本分支无需训练。"""
        return None

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        img_emb = self.encoder.encode_image(image)            # (1, D)
        sim_n = (img_emb @ self._normal_emb.T).mean()
        sim_a = (img_emb @ self._abnormal_emb.T).mean()
        logits = torch.stack([sim_n, sim_a]) / self.temperature
        score = float(F.softmax(logits, dim=0)[1])            # P(异常)
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=None,
                            defect_type=self.defect_type, latency_ms=lat)
