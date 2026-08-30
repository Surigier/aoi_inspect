#!/usr/bin/env bash
# 把 RTX 2070 SUPER 降频成 RTX 2060 的算力代理,测 2500² 延时。
#
# 为什么用降频而不是直接拿 2070S/4060L 的数往下猜:
#   2070S 和 2060 **同为 Turing 架构**(4060L 是 Ada,跨两代不可比)。
#   同架构下,把 FP32 吞吐压到相同量级,得到的延时代理可信度高得多。
#
#   2060  : 1920 核 @ ~1680MHz = 6.45 TFLOPS
#   2070S : 2560 核 @  1875MHz = 9.60 TFLOPS   (1875 是本机实测 boost,不是标称)
#   → 锁 2560 核到 6.45e12/(2*2560) ≈ 1260MHz,吞吐对齐
#
# **这个代理仍然偏乐观,真 2060 只会更慢**,汇报时必须标注:
#   ①SM 分布不同(2070S 是更多SM跑低频,2060 是更少SM跑高频,后者调度效率略低)
#   ②显存带宽降不下来(2070S 448GB/s vs 2060 336GB/s,-lmc 多数消费卡不支持)
#   ③2060 只有 6GB 显存,2070S 有 8GB——显存压力这条代理不了
#
# 用法(需要 sudo):bash scripts/lat_2060_proxy.sh
# 跑完自动 nvidia-smi -rgc 恢复,中途 Ctrl-C 也会恢复(trap)。

set -uo pipefail
TARGET_MHZ=1260
cd "$(dirname "$0")/.."

restore() { echo; echo "恢复默认频率…"; sudo -n nvidia-smi -rgc >/dev/null 2>&1 || sudo nvidia-smi -rgc; nvidia-smi --query-gpu=clocks.gr --format=csv,noheader; }
trap restore EXIT INT TERM

echo "锁定核心频率到 ${TARGET_MHZ}MHz(2060 算力代理)…"
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -lgc ${TARGET_MHZ},${TARGET_MHZ}
sleep 2
echo "当前: $(nvidia-smi --query-gpu=name,clocks.gr,clocks.max.gr --format=csv,noheader)"
echo

set -a; . ./.env 2>/dev/null; set +a
PYTHONPATH=. HF_HUB_OFFLINE=1 python3 scripts/eval_phone_stitch.py 2>&1 \
  | grep -vE "FutureWarning|torch.load|Retrying|thrown|weights_only|^  sd ="
