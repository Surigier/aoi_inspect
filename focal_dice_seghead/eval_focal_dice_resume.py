"""断点续跑:pcb/phone_battery/cable已经跑完(见_logs/focal_dice_eval.log),这里
只跑剩下的logical:breakfast_box + logical:pushpins,不重新浪费时间。"""
import torch
from focal_dice_seghead.eval_focal_dice import run_one
from global_context.eval_global_branch import prep_loco


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
    ]
    for name, prep in jobs:
        run_one(name, *prep())


if __name__ == "__main__":
    main()
