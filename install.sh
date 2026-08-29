#!/bin/bash
# 一键安装(考官机器) —— 在解压后的交付目录内执行:bash install.sh
set -e
cd "$(dirname "$0")"
echo "== ① Python依赖 =="
python3 -m pip install -r requirements.txt
python3 -c "import torch" 2>/dev/null || {
  echo "!! 未检测到 torch。请按显卡环境安装(示例:CUDA 12.1):"
  echo "   pip install torch --index-url https://download.pytorch.org/whl/cu121"
  exit 1
}
echo "== ② 离线权重自检(评委机器可无外网) =="
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/pack_offline_weights.py --verify
echo "== ③ 单元测试 =="
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m pytest -q -p no:warnings
echo ""
echo "✅ 安装完成。使用方式见 docs/delivery/使用说明.md"
echo "   评测入口: python3 submit.py --normal <正常图目录> --defect <缺陷图目录> --test <测试目录> --out result.csv"
echo "   Web演示台: bash scripts/start_app.sh 7860"
