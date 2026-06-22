import torch
from aoi.tiled import make_tiles, TiledFewShotDetector


def test_make_tiles_covers_full_image():
    img = torch.zeros(3, 2500, 2500)
    tiles, pos = make_tiles(img, tile=512, stride=512)
    assert tiles.shape[1:] == (3, 512, 512)
    # 末块贴边 → 覆盖到右/下边界
    ys = sorted({p[0] for p in pos})
    xs = sorted({p[1] for p in pos})
    assert ys[-1] + 512 == 2500 and xs[-1] + 512 == 2500
    assert len(tiles) == len(ys) * len(xs)


def test_make_tiles_small_image_single_tile():
    img = torch.zeros(3, 300, 300)
    tiles, pos = make_tiles(img, tile=512, stride=512)
    assert len(tiles) == 1 and pos == [(0, 0)]


class _FakeBackbone:
    """轻量假骨干:输出小特征图,免下载真权重,测分块流程逻辑。"""
    def extract(self, x):
        b = x.shape[0]
        return torch.rand(b, 8, 16, 16)


def test_tiled_detector_fit_and_predict():
    det = TiledFewShotDetector(_FakeBackbone(), tile=512, stride=512,
                               coreset_ratio=0.5, feat_grid=8)
    normals = [torch.rand(3, 1024, 1024) for _ in range(4)]
    defects = [torch.rand(3, 1024, 1024) for _ in range(2)]
    thr = det.fit_fewshot(normals, defects)
    assert thr is not None
    out = det.predict(torch.rand(3, 1024, 1024))
    assert set(out) == {"score", "is_defect", "worst_tile", "latency_ms"}
    assert isinstance(out["is_defect"], bool)
    assert out["worst_tile"] in {(0, 0), (0, 512), (512, 0), (512, 512)}
