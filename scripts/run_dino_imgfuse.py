"""DINOv2图级平等融合(最后一角度):max(z_EAD,z_DINO)联合阈值,治EAD漏检。
≠灰区补检(那是兜底,失败);此为平等融合,能抓EAD深度漏检。
量:EAD-only vs 融合的 召回(缺陷检出率)/正常准确率/平衡acc。
用法:python scripts/run_dino_imgfuse.py
"""
import numpy as np
import torch
from scripts.run_dino_rescue import (DinoBank, prep_mvtec, prep_realiad)
from aoi.efficientad import EfficientADDetector
from aoi.fewshot import FewShotAdapter

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def z(v, mu, sd):
    return (np.array(v) - mu) / sd


def run(name, normals, defs, goods):
    fit_n, fit_d = normals[:60], defs[:30]
    test_d, test_g = defs[30:70], goods
    ead = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    ead.fit_fewshot(fit_n, None)
    bank = DinoBank(fit_n[:40])
    en = [ead._image_score(x)[0] for x in fit_n]; dn = [bank.score(x) for x in fit_n]
    emu, esd = np.mean(en), np.std(en) + 1e-9
    dmu, dsd = np.mean(dn), np.std(dn) + 1e-9
    ed = [ead._image_score(x)[0] for x in fit_d]; dd = [bank.score(x) for x in fit_d]
    # 联合阈值:平衡acc标定(用fit正常+缺陷)
    fus_n = np.maximum(z(en, emu, esd), z(dn, dmu, dsd))
    fus_d = np.maximum(z(ed, emu, esd), z(dd, dmu, dsd))
    thr_e = FewShotAdapter._calibrate(en, ed)
    thr_f = FewShotAdapter._calibrate(list(fus_n), list(fus_d))
    # 测试
    e_td = [ead._image_score(x)[0] for x in test_d]; d_td = [bank.score(x) for x in test_d]
    e_tg = [ead._image_score(x)[0] for x in test_g]; d_tg = [bank.score(x) for x in test_g]
    f_td = np.maximum(z(e_td, emu, esd), z(d_td, dmu, dsd))
    f_tg = np.maximum(z(e_tg, emu, esd), z(d_tg, dmu, dsd))
    # EAD-only
    rec_e = np.mean([s >= thr_e for s in e_td]); nacc_e = np.mean([s < thr_e for s in e_tg])
    # 融合
    rec_f = np.mean([s >= thr_f for s in f_td]); nacc_f = np.mean([s < thr_f for s in f_tg])
    print(f"{name:10s} EAD: 召回={rec_e:.3f} 正常acc={nacc_e:.3f} 平衡={(rec_e+nacc_e)/2:.3f}  | "
          f"融合: 召回={rec_f:.3f} 正常acc={nacc_f:.3f} 平衡={(rec_f+nacc_f)/2:.3f}  "
          f"Δ平衡={(rec_f+nacc_f)/2-(rec_e+nacc_e)/2:+.3f}", flush=True)


def main():
    torch.manual_seed(0)
    print("=== DINOv2图级平等融合 max(z_EAD,z_DINO)(治漏检)===")
    for name, prep in [("pcb", lambda: prep_realiad("pcb")),
                       ("battery", lambda: prep_realiad("phone_battery")),
                       ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
                       ("pill", lambda: prep_mvtec("pill", ["color"])),
                       ("cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"]))]:
        run(name, *prep())


if __name__ == "__main__":
    main()
