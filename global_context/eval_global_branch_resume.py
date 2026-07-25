"""断点续跑:上次进程掉线前logical的4/5类(breakfast_box/juice_bottle/pushpins/
screw_bag)已经跑完并记录在_logs/global_context_eval.log,这里只跑剩下的5类
(splicing_connectors逻辑 + 2类结构回归检查 + 2类生产回归检查),结果手动并入
最终报告,不重新浪费时间跑已完成的部分。"""
import torch
from global_context.eval_global_branch import (
    prep_loco, prep_mvtec, prep_realiad, run_one,
)
import numpy as np


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:splicing_connectors", lambda: prep_loco("splicing_connectors", "logical_anomalies")),
        ("structural:breakfast_box(回归检查)", lambda: prep_loco("breakfast_box", "structural_anomalies")),
        ("structural:juice_bottle(回归检查)", lambda: prep_loco("juice_bottle", "structural_anomalies")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("生产:pcb(回归检查)", lambda: prep_realiad("pcb")),
    ]
    for name, prep in jobs:
        run_one(name, *prep())


if __name__ == "__main__":
    main()
