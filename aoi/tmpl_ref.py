"""金模板参考库(工业AOI经典):最近邻挑正常参考 + ECC刚性配准。
供模板差分特征(feat(test) ⊕ feat(test)-feat(aligned_ref)):刚性件微小缺陷在差分里信号强。
实测 pcb IoU +0.052(+21%),battery 中性 → 由 fit 留出集自动决定开关。"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _gray64(img):
    g = img.mean(0)
    return F.interpolate(g[None, None], size=(64, 64), mode="bilinear")[0, 0].cpu().numpy()


class RefBank:
    def __init__(self, normals, max_refs=40):
        self.refs = list(normals[:max_refs])
        self.keys = np.stack([_gray64(n) for n in self.refs])

    def aligned_ref(self, img):
        """最近邻参考 + ECC欧氏配准(失败回退原参考)。"""
        q = _gray64(img)
        i = int(np.argmin(((self.keys - q) ** 2).mean(axis=(1, 2))))
        ref = self.refs[i]
        if not _HAS_CV2:
            return ref
        try:
            g1 = cv2.resize((img.mean(0).cpu().numpy() * 255).astype(np.uint8), (256, 256))
            g2 = cv2.resize((ref.mean(0).cpu().numpy() * 255).astype(np.uint8), (256, 256))
            warp = np.eye(2, 3, dtype=np.float32)
            cv2.findTransformECC(g1, g2, warp, cv2.MOTION_EUCLIDEAN,
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
            H, W = ref.shape[-2:]
            scale = np.diag([W / 256, H / 256, 1]).astype(np.float32)
            w3 = np.vstack([warp, [0, 0, 1]])
            w_full = (scale @ w3 @ np.linalg.inv(scale))[:2]
            arr = ref.permute(1, 2, 0).cpu().numpy()
            out = cv2.warpAffine(arr, w_full, (W, H),
                                 flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            return torch.from_numpy(out).permute(2, 0, 1)
        except Exception:
            return ref
