from pathlib import Path
import torch
import numpy as np
from PIL import Image


def _load_img(path: Path, size: int = 320) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0      # H,W,3
    return torch.from_numpy(arr).permute(2, 0, 1)        # 3,H,W


def load_category(root: str, size: int = 320) -> dict:
    """读取 MVTec 单类别,返回 train_normal / test_normal / test_defect 张量列表。"""
    root = Path(root)
    train_normal = [_load_img(p, size) for p in sorted((root / "train" / "good").glob("*.png"))]
    test_normal = [_load_img(p, size) for p in sorted((root / "test" / "good").glob("*.png"))]
    test_defect = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            test_defect += [_load_img(p, size) for p in sorted(sub.glob("*.png"))]
    return {"train_normal": train_normal, "test_normal": test_normal, "test_defect": test_defect}
