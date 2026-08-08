"""断点续跑:hazelnut/cable/pill/carpet/leather已经跑完(见_logs/seghead_aggressive.
log),这里只跑剩下的metal_nut/wood/pcb/phone_battery/breakfast_box。"""
import torch
from seghead_tuning.probe_aggressive_train import run_one
from global_context.eval_global_branch import prep_loco, prep_realiad
from scripts.run_scorecard_5types import prep_mvtec_color


def main():
    torch.manual_seed(0)
    jobs = [
        ("色彩 metal_nut", lambda: prep_mvtec_color("metal_nut")[:4]),
        ("色彩 wood", lambda: prep_mvtec_color("wood")[:4]),
        ("生产:pcb(微小缺陷)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    for name, prep in jobs:
        run_one(name, *prep())


if __name__ == "__main__":
    main()
