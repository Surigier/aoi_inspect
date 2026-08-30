"""把 timm 骨干权重打进交付包 —— 修一个会导致**零分**的交付风险。

问题:`models/` 目录里只带了 EAD 教师和 MobileSAM,但 WRN50 定位骨干和 DINOv2
图级co-detector 的权重是 **timm 在运行时从 HuggingFace 拉的**。评委机器如果没有外网
(工业/内网环境非常常见),程序会直接抛 LocalEntryNotFoundError 起不来——不是精度
掉一点,是一行都跑不了。这个风险在迁移到2070那台内网机器时被真实触发过。

解法:把 HF 缓存目录整个搬进仓库,运行时用 HF_HUB_CACHE 指过去 + HF_HUB_OFFLINE=1。
已验证:把 HF_HUB_CACHE 指向自带目录后,timm 完全不联网即可加载两个骨干。

用法:
  python scripts/pack_offline_weights.py            # 打包到 models/hf_cache/
  python scripts/pack_offline_weights.py --verify   # 只校验现有包能不能离线加载
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "models" / "hf_cache"
# 生产 locate() 路径上真正会加载的两个 timm 模型(aoi/backbone.py 与 aoi/dino_gate.py)
NEEDED = ["models--timm--wide_resnet50_2.racm_in1k",       # 定位骨干(aoi/backbone.py)
          "models--timm--vit_small_patch14_dinov2.lvd142m"]  # 图级co-detector(aoi/dino_gate.py)
# 零样本CLIP不在默认包里:它只服务 submit.py --zeroshot 这条旁路,而赛题评分协议是
# "100正常+30缺陷"的少样本路径。CLIP权重600MB+,默认打进去会让交付包翻倍。
# 需要无网机器上也能跑零样本,加 --with-clip。
CLIP = "models--timm--vit_base_patch16_clip_224.openai"


def _size(d):
    """真实占盘。不能直接 rglob+stat 求和:HF缓存的 snapshots/ 是指向 blobs/ 的软链,
    跟着软链走会把同一个 blob 算两遍(实测348MB被报成728MB)。"""
    return sum(f.stat().st_size for f in d.rglob("*")
               if not f.is_symlink() and f.is_file())


def _src_cache():
    for c in (os.environ.get("HF_HUB_CACHE"),
              os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
              os.path.expanduser("~/.cache/huggingface/hub")):
        if c and Path(c).is_dir():
            return Path(c)
    return None


def verify():
    """在子进程里用打好的包做一次真离线加载——必须子进程,因为 huggingface_hub
    在 import 时就读掉了环境变量,同进程里改没用。"""
    env = dict(os.environ, HF_HUB_CACHE=str(DST), HF_HUB_OFFLINE="1")
    code = ("import timm;"
            "timm.create_model('wide_resnet50_2.racm_in1k',pretrained=True,features_only=True,out_indices=(1,2));"
            "timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,num_classes=0);"
            "print('OFFLINE_LOAD_OK')")
    # 重试3次:本机的 torch/timm/mpmath **导入阶段**会随机抛 TypeError
    # (如 "'str' object is not callable"、"attribute name must be string"),
    # 与权重包无关,重试即好。不重试会把好包误判成坏包。
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        if "OFFLINE_LOAD_OK" in r.stdout:
            print("离线加载校验: ✅ 通过")
            return True
    print("离线加载校验: ❌ 失败(已重试3次)\n" + r.stderr[-800:])
    return False


def main():
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    needed = list(NEEDED) + ([CLIP] if "--with-clip" in sys.argv else [])
    src = _src_cache()
    if src is None:
        print("找不到本机 HF 缓存,先在联网机器上跑一次让 timm 把权重拉下来"); sys.exit(1)
    DST.mkdir(parents=True, exist_ok=True)
    for name in needed:
        s = src / name
        if not s.is_dir():
            print(f"❌ 本机缓存里没有 {name},先在联网机器上加载一次该模型"); sys.exit(1)
        d = DST / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d, symlinks=True)          # symlinks=True:HF缓存靠 snapshots→blobs 的软链
        print(f"→ {name}  {_size(d)/1e6:.0f} MB")
    total = _size(DST) / 1e6
    print(f"打包完成: {DST}  共 {total:.0f} MB")
    verify()


if __name__ == "__main__":
    main()
