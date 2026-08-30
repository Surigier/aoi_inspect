#!/bin/bash
# 打交付包:代码 + 离线权重 + 文档。明确排除:.env(密钥!)、数据集、实验日志、缓存。
set -e
set -o pipefail   # 管道中间命令失败也要让整个流水线失败——踩过的坑见下方注释
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/aoi_release}"
NAME="aoi_inspect_release"
rm -rf "$OUT/$NAME"; mkdir -p "$OUT/$NAME"
# 代码:按**工作区**打包(不是git HEAD——跨机rsync的工作区可能领先于本地git历史,
# 用git archive曾打出过8月24日的旧包)。ls-files -co --exclude-standard = 已跟踪+
# 未跟踪但不含gitignore(.env/_logs/数据集天然排除)。
#
# 【实测踩过的两个坑,2026-08-30,叠在一起才现形】
# 坑①:git ls-files默认对非ASCII文件名(本仓库里中文目录名demo_data/exam_data
#   到处都是)做八进制转义并加引号,如 "exam_data/\350\200\203...",这段转义后的
#   字符串被当**字面文件名**传给tar -T -,tar当然找不到——报"Cannot stat"。
#   用 -z (NUL分隔、不转义) 配 tar --null 彻底绕开,不依赖core.quotePath配置。
# 坑②:没有 pipefail 时,管道中间的 tar -cf 遇上坑①报错,整个流水线只看最后
#   一条命令(tar -x)的退出码——tar -x对着截断的流不一定报错,于是"打包成功"
#   但2460个应有文件里只落地了288个,静默消失,只有事后逐文件比对才发现。
git ls-files -co --exclude-standard -z | tar --null -T - -cf - | tar -x -C "$OUT/$NAME"
# 完整性核对(第二道防线,双保险):逐文件比对git应有清单与实际落地文件。
missing=$(comm -23 <(git ls-files -co --exclude-standard -z | tr '\0' '\n' | sort) \
                    <(cd "$OUT/$NAME" && find . -type f | sed 's|^\./||' | sort) | wc -l)
if [ "$missing" -ne 0 ]; then
  echo "❌ 打包不完整:应有文件中有 $missing 个未落地,终止(不产出残缺包)"
  comm -23 <(git ls-files -co --exclude-standard -z | tr '\0' '\n' | sort) \
           <(cd "$OUT/$NAME" && find . -type f | sed 's|^\./||' | sort) | head -10
  exit 1
fi
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
# pipefail生效后,grep查无匹配(=没查到密钥,正常情况)会让整条管道判定为失败,
# 必须用 || true 兜底,否则"没查到"这个好消息会被set -e当成脚本出错直接杀掉。
stray=$( { grep -rE "sk-[0-9a-f]{32}" "$OUT/$NAME" -l 2>/dev/null | grep -v "aoi/vlm_type.py" | head -1; } || true )
[ -n "$stray" ] && { echo "❌ 密钥出现在预期之外的文件: $stray,终止"; exit 1; }
[ -f "$OUT/$NAME/.env" ] && { echo "❌ .env混入包内,终止"; exit 1; }
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
du -sh "$OUT/$NAME.tar.gz"
echo "✅ 交付包: $OUT/$NAME.tar.gz  (解压后运行 bash install.sh)"
