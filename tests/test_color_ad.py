import torch
from aoi.branches.color_ad import ColorADBranch, rgb_to_lab


def _normal_batch(n=8):
    torch.manual_seed(0)
    base = torch.rand(3, 64, 64) * 0.5 + 0.25
    return torch.stack([(base + torch.randn(3, 64, 64) * 0.01).clamp(0, 1) for _ in range(n)])


def test_color_fires_on_tint_robust_to_brightness():
    imgs = _normal_batch()
    br = ColorADBranch(grid_size=8)
    br.fit(imgs)
    base = br.infer(imgs[0:1]).score
    darker = (imgs[0] * 0.6).clamp(0, 1).unsqueeze(0)
    tinted = imgs[0].clone(); tinted[0] = (tinted[0] + 0.2).clamp(0, 1)
    s_bright = br.infer(darker).score
    s_color = br.infer(tinted.unsqueeze(0)).score
    assert s_color > 3 * (base + 1e-6)          # 颜色偏移强烈报警
    assert s_bright < s_color                    # 亮度变化远不如颜色


def test_color_fit_idempotent():
    imgs = _normal_batch()
    br = ColorADBranch(grid_size=8)
    br.fit(imgs); s1 = br.infer(imgs[0:1]).score
    br.fit(imgs); s2 = br.infer(imgs[0:1]).score
    assert abs(s1 - s2) < 1e-6


def test_rgb_to_lab_shape_and_lightness():
    rgb = torch.rand(2, 3, 16, 16)
    lab = rgb_to_lab(rgb)
    assert lab.shape == (2, 3, 16, 16)
    white = rgb_to_lab(torch.ones(1, 3, 4, 4))
    assert white[0, 0].mean() > 95          # 白色 L≈100
