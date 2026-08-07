"""断点续跑:hazelnut/cable/pill/carpet/leather已经跑完(见_logs/probe_inp.log),
metal_nut跑到一半时撞上torch权重加载的瞬时性报错(TypeError: 'tuple' object is not
callable,session里已知的偶发性torch/timm内部报错,重试即可通过,和本仓库代码无关),
这里只跑剩下的metal_nut+wood,不重新浪费时间。"""
import torch
from inp_former_probe.probe_inp import build_inp_model, run_one, prep_mvtec_color


def main():
    torch.manual_seed(0)
    print("加载INP-Former官方Few-Shot(k=4) MVTec-AD checkpoint...", flush=True)
    inp_model = build_inp_model()
    print("加载完成", flush=True)

    jobs = [
        ("色彩 metal_nut", lambda: prep_mvtec_color("metal_nut")[:4]),
        ("色彩 wood", lambda: prep_mvtec_color("wood")[:4]),
    ]
    for name, prep in jobs:
        run_one(name, *prep(), inp_model)


if __name__ == "__main__":
    main()
