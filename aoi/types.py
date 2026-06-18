from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class BranchResult:
    """单个检测分支的输出。所有分支统一返回此结构。"""
    score: float                                  # 图像级异常分,越大越异常
    anomaly_map: Optional[np.ndarray] = None      # HxW 像素级异常图
    boxes: List = field(default_factory=list)     # [(x1,y1,x2,y2,label), ...]
    defect_type: str = "unknown"
    latency_ms: float = 0.0
