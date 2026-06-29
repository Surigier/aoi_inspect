"""Real-IAD 泛化基准(最对口手机域:含 phone_battery/pcb/usb/sim_card 等真实电子件)。
官方协议:100 正常 + 30 缺陷现场迁移 → 测剩余。报 fusAU/acc/延时,默认只跑手机相关类。
用法:python scripts/run_realiad_bench.py [all|phone]   (默认 phone 子集)
数据:data/_dl/realiad_jsons/realiad_jsons_sv/<cat>.json + 图片根 IMG_ROOT。
"""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict
import torch
from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from eval.protocol import run_protocol
from eval.mvtec import _load_img

SIZE = 320
JSON_DIR = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
IMG_ROOT = Path("data/_dl/Real-IAD")          # 解压后按类别分子目录;下方自动探测

# 手机/电子件相关(赛题隐藏域:屏/电池/中框)
PHONE = ["phone_battery", "pcb", "usb", "usb_adaptor", "sim_card_set", "button_battery",
         "transistor1", "terminalblock", "audiojack", "switch", "regulator", "end_cap"]


def _resolve(cat, rel):
    """json 的 image_path 相对某层;探测真实前缀(IMG_ROOT/cat 或 IMG_ROOT)。"""
    for base in (IMG_ROOT / cat, IMG_ROOT, IMG_ROOT / "realiad_512" / cat, IMG_ROOT / cat / cat):
        p = base / rel
        if p.exists():
            return p
    return IMG_ROOT / cat / rel          # 缺省


def load_cat(cat):
    d = json.load(open(JSON_DIR / f"{cat}.json"))
    tn, te_n, te_d = [], [], []
    for split, bucket_ok, bucket_ng in [("train", tn, te_d), ("test", te_n, te_d)]:
        for it in d.get(split, []):
            p = _resolve(cat, it["image_path"])
            if not p.exists():
                continue
            img = _load_img(p, SIZE)
            (bucket_ok if it["anomaly_class"] == "OK" else bucket_ng).append(img)
    return {"train_normal": tn, "test_normal": te_n, "test_defect": te_d}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "phone"
    cats = sorted([p.stem for p in JSON_DIR.glob("*.json")]) if mode == "all" else PHONE
    torch.manual_seed(0)
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    agg = [0.0, 0.0, 0.0, 0]
    for cat in cats:
        try:
            data = load_cat(cat)
        except Exception as e:
            print(f"{cat}: skip ({e})"); continue
        if len(data["train_normal"]) == 0 or len(data["test_defect"]) < 6:
            print(f"{cat}: skip (n_norm={len(data['train_normal'])} n_def={len(data['test_defect'])})"); continue
        rng = random.Random(0)
        nm, df = data["train_normal"][:], data["test_defect"][:]
        rng.shuffle(nm); rng.shuffle(df)
        nfit, dfit = min(100, len(nm)), min(30, len(df) // 2)
        fn, fd = nm[:nfit], df[:dfit]
        ti = data["test_normal"] + df[dfit:]
        tl = [0] * len(data["test_normal"]) + [1] * len(df[dfit:])
        try:
            m = run_protocol(default_adapter(bb), fn, fd, ti, tl)
        except Exception as e:
            print(f"{cat}: err ({e})"); continue
        agg[0] += m["auroc"]; agg[1] += m["accuracy"]; agg[2] += m["latency_ms_mean"]; agg[3] += 1
        star = " ⭐手机件" if cat in PHONE else ""
        print(f"{cat:16s} fusAU={m['auroc']:.3f} acc={m['accuracy']:.3f} lat={m['latency_ms_mean']:.0f}ms{star}", flush=True)
    if agg[3]:
        print(f"\n均值({agg[3]}类): fusAU={agg[0]/agg[3]:.3f} acc={agg[1]/agg[3]:.3f} lat={agg[2]/agg[3]:.0f}ms")


if __name__ == "__main__":
    main()
