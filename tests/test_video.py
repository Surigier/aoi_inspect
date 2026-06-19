import torch
from aoi.video import moving_average, group_events, VideoDetector
from aoi.types import BranchResult


def test_moving_average_trailing():
    assert moving_average([0.0, 0.0, 3.0, 3.0], 2) == [0.0, 0.0, 1.5, 3.0]


def test_group_events_filters_short_runs():
    # idx0 单帧(len1)被滤,idx2-3(len2)保留
    assert group_events([True, False, True, True, False], min_len=2) == [(2, 3)]


class _MeanAdapter:
    """分数=帧均值,阈值 0.5。"""
    threshold = 0.5

    def predict(self, frame):
        s = float(frame.mean())
        return BranchResult(score=s), s >= self.threshold


def _frame(v):
    return torch.full((3, 4, 4), v)


def test_detects_defect_event():
    frames = [_frame(0.1)] * 3 + [_frame(0.9)] * 3   # 正常后接缺陷段
    out = VideoDetector(_MeanAdapter(), smooth_window=2, min_event_len=2).process(frames)
    assert len(out["frame_scores"]) == 6
    assert len(out["events"]) == 1
    start, end = out["events"][0]
    assert start >= 3 and end == 5


def test_single_frame_spike_debounced():
    frames = [_frame(0.1)] * 3 + [_frame(0.9)] + [_frame(0.1)] * 3   # 单帧尖峰
    out = VideoDetector(_MeanAdapter(), smooth_window=1, min_event_len=2).process(frames)
    assert out["events"] == []
