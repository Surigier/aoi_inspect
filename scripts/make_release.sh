#!/bin/bash
# 打交付包:代码 + 离线权重 + 文档。明确排除:.env(密钥!)、数据集、实验日志、缓存。
set -e
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/aoi_release}"
NAME="aoi_inspect_release"
rm -rf "$OUT/$NAME"; mkdir -p "$OUT/$NAME"
# 代码(走git清单,天然排除gitignore里的.env/_logs/数据)
git archive HEAD | tar -x -C "$OUT/$NAME"
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
# 安全闸门:包里绝不能有密钥
if grep -rE "sk-[0-9a-f]{32}" "$OUT/$NAME" --include="*" -l 2>/dev/null | head -1 | grep -q .; then
  echo "❌ 发现疑似密钥,终止打包"; exit 1
fi
[ -f "$OUT/$NAME/.env" ] && { echo "❌ .env混入包内,终止"; exit 1; }
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
du -sh "$OUT/$NAME.tar.gz"
echo "✅ 交付包: $OUT/$NAME.tar.gz  (解压后运行 bash install.sh)"
