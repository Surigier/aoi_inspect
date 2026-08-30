#!/bin/bash
# 一键安装(考官机器) —— 在解压后的交付目录内执行:bash install.sh
set -e
cd "$(dirname "$0")"
mkdir -p _logs
echo "== ① Python依赖 =="
# 整体安装失败时(常见原因:某个包所在的镜像临时抽风,与依赖内容无关)逐包重试,
# 跳过装不上的——requirements.txt里标"缺→回退/静默降级"的包本来就是可选加速项,
# 代码里已有try/except兜底,不该因为一个包卡住其余全部依赖。
python3 -m pip install -r requirements.txt || {
  echo "!! 整体安装失败,逐包重试(跳过装不上的可选依赖,不影响能装上的部分)"
  fail=""
  while IFS= read -r line; do
    pkg=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$pkg" ] && continue
    python3 -m pip install "$pkg" || fail="$fail $pkg"
  done < requirements.txt
  [ -n "$fail" ] && echo "!! 以下依赖未能安装,若属于requirements.txt中标注了回退行为的可选项可忽略,否则手动安装:$fail"
}
python3 -c "import torch" 2>/dev/null || {
  echo "!! 未检测到 torch。请按显卡环境安装(示例:CUDA 12.1):"
  echo "   pip install torch --index-url https://download.pytorch.org/whl/cu121"
  exit 1
}
echo "== ② 离线权重自检(评委机器可无外网) =="
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/pack_offline_weights.py --verify
echo "== ③ 单元测试 =="
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m pytest -q -p no:warnings
python3 -c "import sys; print(sys.executable)" > _logs/python_bin.txt 2>/dev/null || true
echo ""
echo "✅ 安装完成。使用方式见 docs/delivery/使用说明.md"
echo "   评测入口: python3 submit.py --normal <正常图目录> --defect <缺陷图目录> --test <测试目录> --out result.csv"
echo "   Web演示台: bash scripts/start_app.sh 7860"
