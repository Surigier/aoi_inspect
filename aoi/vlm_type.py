"""VLM辅助的缺陷类型标注(只在**fit阶段**调用,推理期零成本)。

为什么放在fit阶段:赛题原文"测试集运行前,仅使用100张带标注无缺陷图片与30张有缺陷
图片进行迁移学习"——**fit不计时**。而locate()要在200ms内返回,一次API往返500ms~2s
直接爆预算,且判分机器可能无外网。所以让VLM在fit阶段把那30张缺陷图的类型标注出来,
用标签训一个极小的最近质心分类器,推理期只跑分类器(微秒级、零API依赖)。

降级路径(必须):VLM不可用(无key/无网/超时/返回不可解析)时返回None,调用方自动
退回原有的启发式类型归属,只是精度低一些,**绝不能让整个检测崩掉**。

key默认内置(leon 2026-08-29拍板:评委就用他的key,开箱即用不需任何配置);
环境变量 DASHSCOPE_API_KEY 可覆盖。赛后应到DashScope控制台轮换此key。
"""
import base64
import io
import json
import os
import urllib.error
import urllib.request

import numpy as np

#: 内置默认key(见模块docstring;env可覆盖;赛后轮换)
_DEFAULT_KEY = "sk-627310a74e454dea99340f3cc40002f3"

# 赛题原文的5类缺陷(顺序即分类器的类别顺序)
TYPES = ["尺寸偏差", "缺件少件", "逻辑错误", "色彩变化", "常见外观缺陷"]
_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max")
# 提示词经过一轮实测优化:初版把"污渍"写进色彩变化的说明里,与"常见外观缺陷"的
# "表面损伤"语义重叠,导致色彩类18次错判里14次倒向"常见外观缺陷"。改为按
# **物理成因**区分(有没有材料被破坏),并显式给出优先判据。
_PROMPT = (
    "你是工业AOI质检专家。图中红框标出了自动检测系统定位到的缺陷区域。"
    "请判断这处缺陷属于以下哪一类,只回答类别名称本身,不要解释:\n"
    + "\n".join(f"- {t}" for t in TYPES)
    + "\n\n判断要点(按物理成因区分,关键看**材料有没有被破坏**):\n"
    "- 色彩变化:材料完整、表面平整,只是**颜色/色泽**和周围或标准不同"
    "(变色、色差、偏色、氧化发黄、印染不均)。没有凹陷、裂口、缺损。\n"
    "- 常见外观缺陷:材料**被破坏**了——划伤、裂纹、凹坑、破损、崩边、毛刺,"
    "能看出表面被划开/压凹/掉块。\n"
    "- 缺件少件:本该存在的部件整个不见了,该位置露出底板/背景。\n"
    "- 逻辑错误:部件都在,但位置/朝向/顺序/装配关系不对。\n"
    "- 尺寸偏差:部件在、也没坏,但尺寸/轮廓比标准偏大或偏小。\n\n"
    "若只是颜色不同、材料没有破损,一律判**色彩变化**,不要判成常见外观缺陷。\n"
    "但有一个例外:如果该位置**整个部件都不见了**、露出了底板或背景,"
    "那属于**缺件少件**——哪怕露出的底板颜色和周围明显不同,也不要判成色彩变化。"
)


_PROMPT_PAIR = _PROMPT.replace(
    "图中红框标出了自动检测系统定位到的缺陷区域。",
    "给你两张图:**第一张是同一产品的正常参考图**,**第二张是待判图**"
    "(红框标出了自动检测系统定位到的缺陷区域)。请**对比这两张图**,"
    "看红框位置相对正常参考发生了什么变化,再判断缺陷类别。"
)


def _crop_at(img, box, out=448):
    """按给定坐标裁正常参考图——必须和缺陷图裁同一块区域,VLM才能做逐位置对比。"""
    from PIL import Image
    cy0, cy1, cx0, cx1 = box
    x = img if img.dim() == 3 else img[0]
    arr = (x[:, cy0:cy1, cx0:cx1].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    im = Image.fromarray(arr)
    sc = out / max(im.size)
    im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue()


def _crop_with_box(img, mask, pad=0.6, out=448):
    """按掩膜连通域裁出缺陷区域(带上下文),并画红框标出缺陷位置 → PNG字节。
    给VLM带上下文很重要:只给缺陷小块的话,VLM看不出"周围本该是什么",
    没法区分"缺件"和"表面损伤"。"""
    from PIL import Image, ImageDraw
    import torch.nn.functional as F
    import torch

    x = img if img.dim() == 3 else img[0]
    H, W = x.shape[-2:]
    m = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                      size=(H, W), mode="nearest")[0, 0].numpy() > 0.5
    if not m.any():
        return None, None
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    cy0 = max(0, int(y0 - h * pad)); cy1 = min(H, int(y1 + h * pad) + 1)
    cx0 = max(0, int(x0 - w * pad)); cx1 = min(W, int(x1 + w * pad) + 1)
    arr = (x[:, cy0:cy1, cx0:cx1].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    im = Image.fromarray(arr)
    sc = out / max(im.size)
    im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([( (x0 - cx0) * sc, (y0 - cy0) * sc), ((x1 - cx0) * sc, (y1 - cy0) * sc)],
                outline=(255, 0, 0), width=max(2, int(3 * sc)))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue(), (cy0, cy1, cx0, cx1)


def _ask(png_bytes, api_key, timeout=30, ref_png=None):
    """问一次VLM,返回类别名或None(任何异常都吞掉→调用方降级)。
    ref_png不为空时走**双图对比**:第一张正常参考、第二张待判图。
    这对"缺件少件/尺寸偏差/逻辑错误"是决定性的——这三类本质上是**比较**出来的,
    没有参照物VLM根本无法判断"这里本该有个东西"(实测cable的缺线被判成色彩变化,
    就是因为它只看到"这块颜色不一样",不知道那里本该有根线)。"""
    def _img(b):
        return {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(b).decode()}"}}
    content = ([_img(ref_png)] if ref_png else []) + [_img(png_bytes),
                                                       {"type": "text", "text": _PROMPT_PAIR if ref_png else _PROMPT}]
    body = {"model": _MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 32, "temperature": 0.0}
    req = urllib.request.Request(
        _ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception:
        return None
    for t in TYPES:                                   # 宽松匹配:回答里包含哪个类别名就算哪个
        if t in txt:
            return t
    return None


_PROMPT_DIAG = (
    "你是工业AOI质检专家。给你两张图:**第一张是同一产品的正常参考图**,"
    "**第二张是待判图**(红框标出操作员反馈的缺陷位置)。\n"
    "请对比两张图,用**一句话(40字以内)**说明红框处相对正常参考发生了什么物理变化,"
    "然后另起一行给出类别名(只写类别名本身)。类别只能是以下之一:\n"
    + "\n".join(f"- {t}" for t in TYPES) +
    "\n\n输出格式严格如下,不要多余内容:\n现象:<一句话>\n类别:<类别名>"
)


def diagnose_defect(img, mask, normal_ref=None, api_key=None, timeout=30):
    """**操作员反馈时的即时诊断**:说出"这是什么缺陷",而不是只给一个类别名。

    为什么需要:赛题要求"操作员反馈 → 系统可回溯检测逻辑"。`explain()` 回溯的是
    模型内部的分数链路(z分/阈值/各门是否触发)——那是给开发者看的;**操作员看不懂**。
    他想知道的是"这到底是什么缺陷"。本函数用自然语言回答这一句,并同时给出类别标签,
    标签直接进类型头的训练集,形成闭环。

    只在**反馈路径**(冷路径,不计时)调用,不碰推理热路径。
    任何异常都吞掉,返回 (None, None) → 调用方降级为"仅记录样本,不给诊断"。
    """
    try:
        png, box = _crop_with_box(img, mask)
        if png is None:
            return None, None
        ref_png = _crop_at(normal_ref, box) if (normal_ref is not None and box) else None
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or _DEFAULT_KEY or _DEFAULT_KEY
        if not api_key:
            return None, None
        def _img(b):
            return {"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64.b64encode(b).decode()}"}}
        content = ([_img(ref_png)] if ref_png else []) + [_img(png),
                                                          {"type": "text", "text": _PROMPT_DIAG}]
        body = {"model": _MODEL, "messages": [{"role": "user", "content": content}],
                "max_tokens": 128, "temperature": 0.0}
        req = urllib.request.Request(
            _ENDPOINT, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        phen = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("现象"):
                phen = line.split(":", 1)[-1].split(":", 1)[-1].strip()
        typ = next((t for t in TYPES if t in txt), None)
        return phen or txt.strip()[:60], typ
    except Exception:
        return None, None


def label_defect_types(images, masks, api_key=None, verbose=True, normal_ref=None):
    """给fit阶段的缺陷图打类型标签。返回 list[str|None],长度同images;
    全部失败(或无key)时返回None,调用方须降级到启发式。
    normal_ref:一张正常参考图(3,H,W)。传了就走双图对比模式——对
    缺件少件/尺寸偏差/逻辑错误这三类是决定性的(它们本质上是比较出来的)。"""
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or _DEFAULT_KEY
    if not api_key:
        if verbose:
            print("!! VLM类型标注:未设置DASHSCOPE_API_KEY,降级到启发式类型归属", flush=True)
        return None
    labels, n_ok = [], 0
    for i, (img, mk) in enumerate(zip(images, masks)):
        png, box = _crop_with_box(img, mk)
        ref_png = _crop_at(normal_ref, box) if (normal_ref is not None and box) else None
        lab = _ask(png, api_key, ref_png=ref_png) if png is not None else None
        labels.append(lab)
        n_ok += lab is not None
    if verbose:
        from collections import Counter
        print(f"!! VLM类型标注:{n_ok}/{len(images)}张成功 → {dict(Counter(l for l in labels if l))}",
              flush=True)
    return labels if n_ok else None
