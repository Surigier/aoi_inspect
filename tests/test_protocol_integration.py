# tests/test_protocol_integration.py
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol

def test_run_protocol_separates_synthetic(synth_dataset):
    normals, defects = synth_dataset["normal"], synth_dataset["defect"]
    branch = TextureADBranch(backbone=Backbone(pretrained=False), coreset_ratio=1.0)
    adapter = FewShotAdapter(branch)
    test_imgs = normals[10:] + defects[10:]
    test_labels = [0] * 10 + [1] * 10
    metrics = run_protocol(adapter, normals[:10], defects[:10], test_imgs, test_labels)
    assert metrics["auroc"] > 0.8
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["latency_ms_mean"] >= 0.0
