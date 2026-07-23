"""参考图条件差异通道构造(RDDN思路,供YOLO候选框提议器用):
test图 + 最近正常模板 → ECC配准 → RGB(3) + 差异通道(3:Lab色差/灰度梯度差/局部SSIM残差)
= 6通道输入。与crop_cascade的候选生成器同源(同样是ECC模板残差),区别在于这里的
差异通道只是【喂给监督检测器的输入特征】,不是直接阈值化当候选——真假由YOLO从
30张真缺陷+海量normal-normal负样本学出来,不是无监督阈值拍脑袋。"""
import numpy as np
import cv2


def ecc_warp(gray_test, gray_tmpl, size=256):
    """test→template 的欧氏warp(2x3),失败返回None。输入原尺度灰度uint8。"""
    try:
        g1 = cv2.resize(gray_test, (size, size))
        g2 = cv2.resize(gray_tmpl, (size, size))
        warp = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(g1, g2, warp, cv2.MOTION_EUCLIDEAN,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
        return warp
    except Exception:
        return None


def warp_template_to_test(tmpl_rgb_u8, warp, out_hw):
    """把warp(test→tmpl,256²尺度估的)缩放回原生尺度,再把模板warp到test坐标系。"""
    H, W = out_hw
    if warp is None:
        return cv2.resize(tmpl_rgb_u8, (W, H))
    scale = np.diag([W / 256, H / 256, 1]).astype(np.float32)
    w3 = np.vstack([warp, [0, 0, 1]])
    w_full = (scale @ w3 @ np.linalg.inv(scale))[:2]
    tmpl_full = cv2.resize(tmpl_rgb_u8, (W, H))
    return cv2.warpAffine(tmpl_full, w_full, (W, H),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _local_ssim(gray_a, gray_b, win=7, C1=6.5025, C2=58.5225):
    """局部SSIM图(box filter近似,不依赖skimage)。返回[0,1]的(1-ssim)/2残差(1=差异大)。"""
    a = gray_a.astype(np.float64); b = gray_b.astype(np.float64)
    mu_a = cv2.boxFilter(a, -1, (win, win)); mu_b = cv2.boxFilter(b, -1, (win, win))
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = cv2.boxFilter(a * a, -1, (win, win)) - mu_a2
    sig_b2 = cv2.boxFilter(b * b, -1, (win, win)) - mu_b2
    sig_ab = cv2.boxFilter(a * b, -1, (win, win)) - mu_ab
    ssim = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / ((mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2) + 1e-9)
    return np.clip((1 - ssim) / 2, 0, 1)


def diff_channels_u8(test_rgb_u8, tmpl_aligned_rgb_u8):
    """test/tmpl(已对齐,同尺寸)RGB uint8 → (3,H,W) float32[0,1]:Lab色差/灰度梯度差/局部SSIM残差。"""
    lab_a = cv2.cvtColor(test_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(tmpl_aligned_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_diff = np.linalg.norm(lab_a - lab_b, axis=-1)
    lab_diff = lab_diff / (lab_diff.max() + 1e-6)

    gray_a = cv2.cvtColor(test_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(tmpl_aligned_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gxa = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3); gya = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
    gxb = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3); gyb = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
    grad_diff = np.abs(np.sqrt(gxa ** 2 + gya ** 2) - np.sqrt(gxb ** 2 + gyb ** 2))
    grad_diff = grad_diff / (grad_diff.max() + 1e-6)

    ssim_res = _local_ssim(gray_a, gray_b).astype(np.float32)
    return np.stack([lab_diff, grad_diff, ssim_res], axis=0)


def build_6ch(test_img_chw01, tmpl_img_chw01):
    """test_img/tmpl_img:(3,H,W)[0,1] torch tensor(已同尺寸)。返回(6,H,W)float32[0,1]
    numpy:RGB(test)+差异通道(3)。tmpl先用ECC对齐到test坐标系再算差异。"""
    test_u8 = (test_img_chw01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    tmpl_u8 = (tmpl_img_chw01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    gray_test = cv2.cvtColor(test_u8, cv2.COLOR_RGB2GRAY)
    gray_tmpl = cv2.cvtColor(tmpl_u8, cv2.COLOR_RGB2GRAY)
    warp = ecc_warp(gray_test, gray_tmpl)
    H, W = test_u8.shape[:2]
    tmpl_aligned = warp_template_to_test(tmpl_u8, warp, (H, W))
    diff = diff_channels_u8(test_u8, tmpl_aligned)             # (3,H,W)[0,1]
    rgb = test_img_chw01.cpu().numpy().astype(np.float32)      # (3,H,W)[0,1]
    return np.concatenate([rgb, diff], axis=0)                 # (6,H,W)
