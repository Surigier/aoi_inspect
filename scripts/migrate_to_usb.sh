#!/usr/bin/env bash
# 把这台机器上跑AOI必需的东西拷到U盘 —— 只拷必需的,不是整仓库。
#
# 为什么不整个拷:这台机器上 aoi_inspect 连数据一共 119G,但真正跑得起来只需要 ~13.5G。
# 大头全是用不上的:rddn_yolo(22G,另一个YOLO项目)、mvtec_ad_2(62G,铁律#4只做大图
# 压测而压测已用手机拼接图做过)、各种解压过的原始.zip/.tar(15G,纯重复占地)。
#
# 代码不走U盘走GitHub(仓库干净、~3MB、还是交付物本体)。U盘只带GitHub装不下的:
# 331MB的DINOv2权重超了GitHub单文件100MB上限,以及数据集。
#
# 用法:  bash scripts/migrate_to_usb.sh /media/srj/U盘挂载点
#        bash scripts/migrate_to_usb.sh /mnt/e --min      # 只带Real-IAD,约2.4G

set -euo pipefail
DST="${1:?用法: bash scripts/migrate_to_usb.sh <U盘路径> [--min]}"
MIN="${2:-}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$DST/aoi_payload"

# 必需:权重(GitHub装不下)+ Real-IAD(手机部件成绩单的数据源)
ITEMS=(
  "backbones"                      # DINOv2 331M
  "models"                         # mobile_sam / EAD teachers / wrn  96M
  "data/_dl/Real-IAD"              # 12类目每类~2400张OK图 + 真实像素掩膜  1.8G
  "data/_dl/realiad_jsons"         # prep_realiad 要的 json 索引  110M
  ".env"                           # DASHSCOPE key(**永远不进git**,只在自己机器间走U盘)
)
# 完整档再加:现有成绩单和逻辑异常成绩单要的公开集
if [ "$MIN" != "--min" ]; then
  ITEMS+=(
    "data/mvtec"                   # hazelnut/cable/pill/carpet/metal_nut  5.0G
    "data/_dl/mvtec_loco"          # 逻辑异常成绩单  5.8G
    "data/phone"                   # MSD衍生:类型归属验证用(正常图只有7张,别拿它报精度)
    "data/phone_best"
  )
fi

echo "源: $SRC"
echo "目标: $OUT"
[ "$MIN" = "--min" ] && echo "模式: 精简(只够跑Real-IAD成绩单)" || echo "模式: 完整"
echo
mkdir -p "$OUT"
for it in "${ITEMS[@]}"; do
  if [ ! -e "$SRC/$it" ]; then echo "跳过(不存在): $it"; continue; fi
  echo "→ $it  ($(du -sh "$SRC/$it" | cut -f1))"
  mkdir -p "$OUT/$(dirname "$it")"
  rsync -a --info=progress2 "$SRC/$it" "$OUT/$(dirname "$it")/"
done
echo
echo "完成。总计: $(du -sh "$OUT" | cut -f1)"
cat <<'TIP'

=== 到 2070 那台机器上怎么装回去 ===
  git clone https://github.com/Surigier/aoi_inspect.git && cd aoi_inspect
  rsync -a /媒体/U盘/aoi_payload/ ./          # 权重和数据就位(含.env)
  pip install -r requirements.txt
  python -m pytest -q                          # 应该 62 passed
  PYTHONPATH=. python scripts/eval_phone_stitch.py    # 2070 的 2500² 延时,和 4060L 夹逼估 2060
TIP
