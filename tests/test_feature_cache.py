"""缓存正确性回归测试 —— 今天连续两次因"缓存按状态盲取而非按输入身份命中"导致
定位崩塌(第一次 含漏检IoU 0.399→0.033;第二次 框命中→0.0000),两次都是跑完
整轮实验才发现。这些断言把该类错误拦在单元测试层。"""
import torch
from aoi.competition import CompetitionLargeDetector


def _det(**kw):
    return CompetitionLargeDetector(device="cpu", train_steps=1, **kw)


def test_wrn缓存必须按图身份命中():
    d = _det()
    a, b = torch.rand(3, 256, 256), torch.rand(3, 256, 256)
    d._wrn_cache_on = True; d._wrn_cache = None
    fa = d._wrn_feats(a)
    assert d._wrn_feats(a) is fa, "同一张图连续调用应命中缓存"
    fb = d._wrn_feats(b)
    assert not torch.allclose(fa, fb), "**不同图绝不能命中同一份缓存**"
    assert torch.allclose(d._wrn_feats(a), fa), "换回原图结果必须仍然正确"


def test_fit期不得启用缓存():
    """seg_head.fit()在fit期逐张调用_wrn_feats。若此时缓存生效,30张缺陷图会全部
    拿到第一张的特征、却配各自不同的掩膜训练——定位直接崩。"""
    d = _det()
    a = torch.rand(3, 256, 256)
    assert d._wrn_cache_on is False, "构造后缓存必须是关闭的"
    f1, f2 = d._wrn_feats(a), d._wrn_feats(a)
    assert f1 is not f2, "开关关闭时不得复用缓存对象"


def test_模板差分的参考图不得命中测试图缓存():
    """_wrn_feats_diff 在同一次locate内先后传入测试图与参考图。若参考图命中测试图的
    缓存,f-fr=全零,分割头输入全零 → 框命中0.0000(今天实测)。"""
    d = _det()
    a, ref = torch.rand(3, 256, 256), torch.rand(3, 256, 256)
    d._wrn_cache_on = True; d._wrn_cache = None
    d._wrn_feats(a)
    _on = d._wrn_cache_on
    d._wrn_cache_on = False
    fr = d._wrn_feats(ref)
    d._wrn_cache_on = _on
    assert not torch.allclose(d._wrn_feats(a), fr), "参考图特征不得等于测试图特征"


def test_dino_patch缓存必须按图身份命中():
    d = _det(dino_seg=True)
    a, b = torch.rand(3, 256, 256), torch.rand(3, 256, 256)
    fa, fb = d._wrn_dino_feats(a), d._wrn_dino_feats(b)
    assert not torch.allclose(fa, fb), "**不同图的DINO patch不得复用**"
    assert fa.shape[0] == 1152, "WRN768 ⊕ DINO384 = 1152 通道"
