"""断点续跑:breakfast_box/juice_bottle(logical)/pushpins/screw_bag已经跑完
(见_logs/gcad_aggressive.log),这里跑剩下的splicing_connectors+两个structural
回归检查+cable/pcb回归检查。"""
import torch
from global_context.eval_global_branch import run_one, prep_loco, prep_mvtec, prep_realiad


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
        run_one(name, *prep(), ae_steps=900, ae_lr=3e-3)


if __name__ == "__main__":
    main()
