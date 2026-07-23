"""YOLOv8n双重手术,构造单类"defect"检测器 + 6通道输入:
①nc=1(单类):用ultralytics官方支持的迁移学习方式——从改过nc的yaml构架构、
  加载预训练权重,形状不匹配的检测头分类输出层自动跳过重新初始化,其余(骨干/
  颈部/框回归分支)照常迁移。不是手工拆检测头,用官方机制更稳。
②首层3→6通道:保留预训练RGB权重,新增3个差异通道权重用RGB权重均值初始化
  (更保守可选零初始化),之后6通道input与RGB通道共享同样的下游网络。"""
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import yaml


def make_defect_yolo(weights="yolov8n.pt", extra_ch=3, init="mean"):
    """返回已完成双重手术的YOLO对象:model.model前向接受(B,3+extra_ch,H,W),
    检测头输出对应单类"defect"。"""
    import ultralytics
    from ultralytics import YOLO

    pkg_dir = Path(ultralytics.__file__).parent
    src_yaml = pkg_dir / "cfg" / "models" / "v8" / "yolov8.yaml"
    cfg = yaml.safe_load(open(src_yaml))
    cfg["nc"] = 1
    tmp = Path(tempfile.mkdtemp()) / "yolov8n_defect.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)

    m = YOLO(str(tmp))
    m.load(weights)                                            # 迁移预训练(检测头分类层跳过)

    old_conv = m.model.model[0].conv                            # Conv2d(3,16,k,s,p,bias=False)
    out_ch, in_ch, kh, kw = old_conv.weight.shape
    assert in_ch == 3, f"预期首层输入3通道,实际{in_ch}"
    new_conv = nn.Conv2d(in_ch + extra_ch, out_ch, kernel_size=(kh, kw),
                        stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :in_ch] = old_conv.weight
        if init == "zero":
            new_conv.weight[:, in_ch:] = 0.0
        else:
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, in_ch:] = mean_w.expand(-1, extra_ch, -1, -1)
    m.model.model[0].conv = new_conv
    if hasattr(m.model, "yaml"):
        m.model.yaml["ch"] = in_ch + extra_ch
    # DetectionModel.loss()经criterion(v8DetectionLoss)读self.hyp.box/.cls/.dfl(属性访问,
    # 不能是普通dict)——用YOLOv8官方默认增益(box=7.5/cls=0.5/dfl=1.5)。
    import types
    m.model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    return m


def smoke_forward(m, size=640, in_ch=6):
    x = torch.rand(1, in_ch, size, size)
    m.model.eval()
    with torch.no_grad():
        out = m.model(x)
    return out
