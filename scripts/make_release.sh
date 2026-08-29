#!/bin/bash
# 打交付包:代码 + 离线权重 + 文档。明确排除:.env(密钥!)、数据集、实验日志、缓存。
set -e
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/aoi_release}"
NAME="aoi_inspect_release"
rm -rf "$OUT/$NAME"; mkdir -p "$OUT/$NAME"
# 代码:按**工作区**打包(不是git HEAD——跨机rsync的工作区可能领先于本地git历史,
# 用git archive曾打出过8月24日的旧包)。ls-files -co --exclude-standard = 已跟踪+
# 未跟踪但不含gitignore(.env/_logs/数据集天然排除)。
git ls-files -co --exclude-standard | tar -T - -cf - | tar -x -C "$OUT/$NAME"
# 离线权重三件套
for d in models backbones; do
  [ -d "$d" ] && rsync -a --exclude "*.zip" "$d" "$OUT/$NAME/"
done
# HF缓存(timm WRN50/DINOv2,评委机无外网时必需)
HFC="$OUT/$NAME/models/hf_cache"
mkdir -p "$HFC"
for src in models/hf_cache "$HOME/.cache/huggingface/hub"; do
  [ -d "$src" ] && rsync -a --copy-links "$src"/models--timm--* "$HFC/" 2>/dev/null || true
done
# 安全闸门:.env绝不能进包(内置key是leon 2026-08-29拍板的例外:评委开箱即用,
# 只允许出现在 aoi/vlm_type.py 这一处;出现在其他任何文件都终止)
stray=$(grep -rE "sk-[0-9a-f]{32}" "$OUT/$NAME" -l 2>/dev/null | grep -v "aoi/vlm_type.py" | head -1)
[ -n "$stray" ] && { echo "❌ 密钥出现在预期之外的文件: $stray,终止"; exit 1; }
[ -f "$OUT/$NAME/.env" ] && { echo "❌ .env混入包内,终止"; exit 1; }
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
du -sh "$OUT/$NAME.tar.gz"
echo "✅ 交付包: $OUT/$NAME.tar.gz  (解压后运行 bash install.sh)"
