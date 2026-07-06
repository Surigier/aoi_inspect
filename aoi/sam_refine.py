"""SAM 边界精化(粗到细定位,Segment Any Anomaly 同族思路):
分割头粗掩膜 → 连通域框提示 → MobileSAM 出锐利边界。
实测逐图IoU:pcb +43% / phone_battery +50% / pill +32% / hazelnut +6%(均值0.285→0.351)。
防爆:SAM 掩膜面积>提示框4倍(分割了整个物体而非缺陷)或为空 → 回退原区域掩膜。
开销 ~19-38ms(imgsz=512),仅判缺陷图触发。SAM 不可用时优雅回退原掩膜。"""
from pathlib import Path
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt"


class SamRefiner:
    def __init__(self, imgsz=512):
        self.imgsz = imgsz
        self._m = None
        self._failed = False

    def _model(self):
        if self._m is None and not self._failed:
            try:
                from ultralytics import SAM
                self._m = SAM(str(_WEIGHTS) if _WEIGHTS.exists() else "mobile_sam.pt")
            except Exception:
                self._failed = True
        return self._m

    def refine(self, img_chw, raw_mask):
        """img(3,H,W)[0,1] tensor(CPU) + 二值掩膜(h,w) → 精化掩膜(h,w)。失败回退原掩膜。"""
        m = self._model()
        if m is None or not _HAS_CV2:
            return raw_mask
        H, W = raw_mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return raw_mask
        arr = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ih, iw = arr.shape[:2]
        sx, sy = iw / W, ih / H
        boxes = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 3:
                continue
            px, py = max(2, int(w * 0.15)), max(2, int(h * 0.15))
            boxes.append([max(0, (x - px) * sx), max(0, (y - py) * sy),
                          min(iw, (x + w + px) * sx), min(ih, (y + h + py) * sy)])
        if not boxes:
            return raw_mask
        try:
            r = m.predict(arr, bboxes=boxes, imgsz=self.imgsz, verbose=False)[0]
        except Exception:
            return raw_mask
        if r.masks is None:
            return raw_mask
        out = np.zeros((H, W), np.uint8)
        n_masks = r.masks.data.shape[0]                       # SAM 可能返回少于提示框数
        for k, b in enumerate(boxes):
            bx = [int(b[0] / sx), int(b[1] / sy), int(b[2] / sx), int(b[3] / sy)]
            if k >= n_masks:                                  # 该框无对应掩膜 → 回退原区域
                out[bx[1]:bx[3], bx[0]:bx[2]] |= raw_mask[bx[1]:bx[3], bx[0]:bx[2]]
                continue
            mk = r.masks.data[k].cpu().numpy().astype(np.uint8)
            mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            box_area = max(1, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            if mk.sum() > 4 * box_area or mk.sum() == 0:
                out[bx[1]:bx[3], bx[0]:bx[2]] |= raw_mask[bx[1]:bx[3], bx[0]:bx[2]]
            else:
                out |= mk
        return out
