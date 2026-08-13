"""把4张图拼成一张2×2的2500×2500大图,模拟论坛里提到的"四宫格拼接"猜测,更贴近
真实赛制可能的输入形态(不是简单放大插值,是真拼接)。每张源图缩放到1250×1250后
按2×2摆放;真值掩膜按同样的摆放方式拼接,保证位置对得上。

用法:
    from tile_2500.prep_tiled import prep_tiled
    normals, fit_i, fit_m, test_defs = prep_tiled("hazelnut", ["crack","cut","hole"])
"""
import random
import torch
import torch.nn.functional as F
from global_context.eval_global_branch import prep_mvtec, prep_realiad, prep_loco

TILE = 1250  # 2×2拼出2500×2500


def _resize(img, size=TILE):
    return F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]


def _resize_mask(mask, size=TILE):
    import numpy as np
    t = torch.from_numpy(mask.astype("float32"))[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")[0, 0]
    return (t.numpy() > 0.5).astype(np.uint8)


def _tile4(imgs):
    """imgs: 4张(3,H,W)张量列表(先各自resize到TILE) -> 一张(3,2*TILE,2*TILE)。"""
    top = torch.cat([imgs[0], imgs[1]], dim=2)
    bot = torch.cat([imgs[2], imgs[3]], dim=2)
    return torch.cat([top, bot], dim=1)


def _tile4_masks(masks):
    import numpy as np
    top = np.concatenate([masks[0], masks[1]], axis=1)
    bot = np.concatenate([masks[2], masks[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


def _blank_mask(size=TILE):
    import numpy as np
    return np.zeros((size, size), dtype=np.uint8)


def _compose_normals(pool, seed=0):
    """把正常图池子每4张拼一组,拼不满的丢弃余数。"""
    rng = random.Random(seed)
    idx = list(range(len(pool))); rng.shuffle(idx)
    groups = [idx[i:i + 4] for i in range(0, len(idx) - len(idx) % 4, 4)]
    return [_tile4([_resize(pool[i]) for i in g]) for g in groups]


def _compose_defects(defect_imgs, defect_masks, normal_pool, seed=0):
    """每个"拼接样本"=1张真实缺陷图(随机摆在四宫格某一格)+3张正常图(其余三格),
    真值掩膜只在缺陷所在的那一格非零,其余三格全0——贴近"拼接大图里某一象限
    真实出问题"这个场景。返回(拼接图列表, 拼接掩膜列表)。"""
    rng = random.Random(seed)
    out_imgs, out_masks = [], []
    for img, mask in zip(defect_imgs, defect_masks):
        pos = rng.randrange(4)
        fillers = rng.sample(range(len(normal_pool)), 3)
        tiles_img = [None] * 4
        tiles_mask = [None] * 4
        tiles_img[pos] = _resize(img)
        tiles_mask[pos] = _resize_mask(mask)
        fi = 0
        for i in range(4):
            if tiles_img[i] is None:
                tiles_img[i] = _resize(normal_pool[fillers[fi]])
                tiles_mask[i] = _blank_mask()
                fi += 1
        out_imgs.append(_tile4(tiles_img))
        out_masks.append(_tile4_masks(tiles_mask))
    return out_imgs, out_masks


def prep_tiled(kind, cat, *args, n_norm_pool=40, seed=0):
    """kind: 'mvtec'/'realiad'/'loco'。*args按对应prep_*函数的位置参数传
    (如mvtec传folders,loco传anomaly_type)。返回拼接版(normals, fit_i, fit_m, test_defs)。"""
    if kind == "mvtec":
        normals, fit_i, fit_m, test_defs = prep_mvtec(cat, *args)
    elif kind == "realiad":
        normals, fit_i, fit_m, test_defs = prep_realiad(cat, *args)
    elif kind == "loco":
        normals, fit_i, fit_m, test_defs = prep_loco(cat, *args)
    else:
        raise ValueError(kind)

    normal_pool = normals[:n_norm_pool]
    tiled_normals = _compose_normals(normal_pool, seed=seed)

    fit_defect_imgs = [x for x, _ in [(fit_i[i], fit_m[i]) for i in range(len(fit_i))]]
    tiled_fit_i, tiled_fit_m = _compose_defects(fit_i, fit_m, normal_pool, seed=seed + 1)

    test_imgs = [im for im, _ in test_defs]
    test_masks = [mk for _, mk in test_defs]
    tiled_test_i, tiled_test_m = _compose_defects(test_imgs, test_masks, normal_pool, seed=seed + 2)
    tiled_test_defs = list(zip(tiled_test_i, tiled_test_m))

    return tiled_normals, tiled_fit_i, tiled_fit_m, tiled_test_defs
