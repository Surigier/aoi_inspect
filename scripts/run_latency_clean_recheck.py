"""冷卡复核:run_pareto_scan.py测出"32组合无一<190ms"(最轻档p90仍233ms),但当时GPU
71°C/SM clock 375MHz(满载3105MHz的12%)——同2026-07-07"脏卡假象"(TIME≈ELAPSED单核死磕)
同一模式。等GPU降温后用同样6探针+同样measure_latency()逻辑复测少数关键组合(不需要真实
精度,延时不依赖类别内容),确认是否降温后回到historical 6/6达标水平。
用法:PYTHONPATH=. python scripts/run_latency_clean_recheck.py
"""
import glob
import tempfile
from pathlib import Path
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from scripts.run_pareto_scan import prep_probe_files, measure_latency, _toggle, AD2, PKU

CHECK = [
    (1, False, False, 700_000),   # 最轻档(pareto扫描里p90=233ms,声称超线)
    (1, True, True, 700_000),     # 全开(最重,pareto扫描p90=297ms)
    (2, True, True, 1_100_000),   # 历史记忆里的默认档
]


def main():
    torch.manual_seed(0)
    probe_files = prep_probe_files(tempfile.mkdtemp(prefix="clean_probe_"))
    import subprocess
    t = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,memory.used",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"GPU状态: {t}", flush=True)

    normals = [torch.rand(3, 640, 640) for _ in range(5)]   # 延时基准不依赖内容,合成即可(架构决定延时)
    defects = [torch.rand(3, 640, 640) for _ in range(5)]
    # run_pareto_scan.py漏开compile_infer(submit.py竞赛入口默认True,已验-11~24%)——方法论
    # 不对齐生产,之前"无组合<190ms"结论需在compile_infer=True下复核才算数。
    det = CompetitionLargeDetector(ead_students=2, train_steps=100, compile_infer=True)
    det.fit_fewshot(normals, defects)
    print("fit done (throwaway, 延时无关精度; compile_infer=True对齐submit.py生产配置)", flush=True)

    for students, dino_on, sam_on, mp in CHECK:
        restore = _toggle(det, students, dino_on, sam_on, mp)
        mean_ms, p90_ms = measure_latency(det, list(probe_files.values()))
        restore()
        print(f"  s={students} dino={dino_on!s:5s} sam={sam_on!s:5s} mp={mp//1000:4d}k  "
              f"均值={mean_ms:.0f}ms p90={p90_ms:.0f}ms", flush=True)
    t2 = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"GPU状态(结束时): {t2}", flush=True)


if __name__ == "__main__":
    main()
