"""断点续跑:hazelnut/cable已跑完(见_logs/joint_steps.log),只跑剩下4类。"""
import numpy as np
import torch
from seghead_tuning.probe_joint_steps import run_one, STEP_CANDIDATES
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad


def main():
    torch.manual_seed(0)
    jobs = [
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("生产:pcb", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    names, rows = [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, test_defs = prep()
        r = run_one(name, normals[:-20], fit_i, fit_m, test_defs, normals[-20:])
        if r and 300 in r:
            names.append(name); rows.append(r)
    # 合并已跑完的hazelnut/cable(硬编码自_logs/joint_steps.log,避免重跑35分钟fit)
    done = {"外观 hazelnut": {300: (0.643, 0.667, 0.946), 600: (0.591, 0.514, 0.946), 900: (0.630, 0.620, 0.946)},
            "缺件 cable":   {300: (0.812, 0.933, 0.914), 600: (0.785, 0.900, 0.914), 900: (0.800, 0.933, 0.914)}}
    names = list(done) + names
    rows = list(done.values()) + rows
    for st in STEP_CANDIDATES[1:]:
        di = [r[st][0] - r[300][0] for r in rows if st in r]
        dh = [r[st][1] - r[300][1] for r in rows if st in r]
        da = [r[st][2] - r[300][2] for r in rows if st in r]
        a = np.array(di)
        ok = (np.median(a) >= 0.005 and np.mean(a) > 0
              and (a > 0).sum() >= max(1, len(a) // 2 + 1)
              and np.min(a) >= -0.01 and np.min(np.array(da)) >= -0.01)
        print(f"\n=== steps={st} vs 300(生产) n={len(a)} === ΔIoU median={np.median(a):+.3f} "
              f"mean={np.mean(a):+.3f} min={np.min(a):+.3f} | Δhit mean={np.mean(dh):+.3f} "
              f"| Δacc min={np.min(da):+.3f} | {'通过' if ok else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in a])), flush=True)


if __name__ == "__main__":
    main()
