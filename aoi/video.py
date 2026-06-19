"""视频路径:逐帧检测 + 时序平滑 + 事件聚合(早期拦截)。
核心逻辑无第三方依赖;读视频文件用懒加载 cv2。"""
import torch


def moving_average(xs, window: int):
    """因果(trailing)滑动平均,window>=1。降低逐帧分数抖动。"""
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - window + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


def group_events(flags, min_len: int = 1):
    """连续 True 段聚成事件 [(start, end_inclusive), ...],仅保留长度 >= min_len 的(去抖)。"""
    events = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                events.append((i, j))
            i = j + 1
        else:
            i += 1
    return events


class VideoDetector:
    """包一个已 fit 的 adapter(具备 .predict 与 .threshold):
    逐帧打分 → 时序平滑 → 阈值 → 事件聚合。事件起始帧即"早期拦截"点。"""

    def __init__(self, adapter, smooth_window: int = 3, min_event_len: int = 2):
        self.adapter = adapter
        self.smooth_window = smooth_window
        self.min_event_len = min_event_len

    def process(self, frames) -> dict:
        """frames: list of (3,H,W) [0,1] 张量。返回逐帧分/平滑分/帧标志/事件。"""
        scores = [self.adapter.predict(f.unsqueeze(0))[0].score for f in frames]
        smoothed = moving_average(scores, self.smooth_window)
        thr = self.adapter.threshold
        flags = [s >= thr for s in smoothed]
        events = group_events(flags, self.min_event_len)
        return {
            "frame_scores": scores,
            "smoothed_scores": smoothed,
            "frame_flags": flags,
            "events": events,
        }


def read_video_frames(path: str, size: int = 320, stride: int = 1):
    """读视频为 (3,H,W) [0,1] 张量列表(每 stride 帧取一帧)。需要 opencv-python(懒加载)。"""
    import cv2
    cap = cv2.VideoCapture(path)
    frames, idx = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.astype("float32") / 255.0).permute(2, 0, 1))
        idx += 1
    cap.release()
    return frames
