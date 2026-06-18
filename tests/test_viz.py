import torch
import numpy as np
from PIL import Image
from aoi.viz import overlay_heatmap

def test_overlay_returns_image_matching_input_size():
    img = torch.full((3, 32, 32), 0.5)
    amap = np.zeros((4, 4), dtype=float)
    amap[0, 0] = 1.0
    out = overlay_heatmap(img, amap)
    assert isinstance(out, Image.Image)
    assert out.size == (32, 32)          # (W, H)
    assert out.mode == "RGB"
