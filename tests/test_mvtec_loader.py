from PIL import Image
import numpy as np
from eval.mvtec import load_category


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(path)


def test_load_category_splits(tmp_path):
    root = tmp_path / "mvtec" / "bottle"
    for i in range(5): _img(root / "train" / "good" / f"{i}.png")
    for i in range(3): _img(root / "test" / "good" / f"{i}.png")
    for i in range(4): _img(root / "test" / "broken" / f"{i}.png")
    data = load_category(str(root), size=64)
    assert len(data["train_normal"]) == 5
    assert len(data["test_normal"]) == 3
    assert len(data["test_defect"]) == 4
    assert data["train_normal"][0].shape == (3, 64, 64)   # CHW float[0,1] tensor
