"""判定cable不稳定性的真正来源:是"同进程内排第几个训练"(执行顺序/随机数流位置),
还是GPU浮点本身不确定(cuDNN卷积算法选择/反向传播并行归约顺序,即使种子号和执行
上下文完全相同,每次训练也会有微小数值差异,几千步SGD后累积放大)?
两个完全独立的进程,一样的代码/种子/上下文(cable都是各自进程里第一个也是唯一训练的
类),若结果仍不同→GPU非确定性是主因,不是执行顺序。
用法:PYTHONPATH=. python scripts/diag_determinism.py
"""
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_mvtec, evaluate

torch.manual_seed(0)
normals, fit_i, fit_m, test_defs, goods = prep_mvtec("cable", ["missing_cable", "missing_wire"])
evaluate("cable#determinism_check", normals, fit_i, fit_m, test_defs, goods)
