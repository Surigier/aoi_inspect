"""EfficientAD 学生-教师检测器(无记忆库 → 推理时间恒定,与正常图数量无关)。
移植自官方参考实现 nelson1425/EfficientAD(WACV2024),用打包的蒸馏教师权重。
教师=冻结 PDN;学生在正常图上学模仿教师;学生模仿不了处=异常。
全局 autoencoder 抓逻辑异常。治本项目"记忆库随100张正常图涨→延时爆"的地基病。"""
import time
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fewshot import FewShotAdapter

OUT = 384
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_WEIGHTS = Path(__file__).resolve().parent.parent / "models"


def get_autoencoder(out_channels=OUT):
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
        nn.Conv2d(64, 64, 8),
        nn.Upsample(size=3, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=8, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=15, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=32, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=63, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=127, mode="bilinear"), nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(True), nn.Dropout(0.2),
        nn.Upsample(size=56, mode="bilinear"), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(64, out_channels, 3, 1, 1),
    )


def get_pdn_small(out_channels=OUT, padding=False):
    p = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(3, 128, 4, padding=3 * p), nn.ReLU(True), nn.AvgPool2d(2, 2, 1 * p),
        nn.Conv2d(128, 256, 4, padding=3 * p), nn.ReLU(True), nn.AvgPool2d(2, 2, 1 * p),
        nn.Conv2d(256, 256, 3, padding=1 * p), nn.ReLU(True),
        nn.Conv2d(256, out_channels, 4),
    )


def get_pdn_medium(out_channels=OUT, padding=False):
    p = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(3, 256, 4, padding=3 * p), nn.ReLU(True), nn.AvgPool2d(2, 2, 1 * p),
        nn.Conv2d(256, 512, 4, padding=3 * p), nn.ReLU(True), nn.AvgPool2d(2, 2, 1 * p),
        nn.Conv2d(512, 512, 1), nn.ReLU(True),
        nn.Conv2d(512, 512, 3, padding=1 * p), nn.ReLU(True),
        nn.Conv2d(512, out_channels, 4), nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 1),
    )


class EfficientADDetector:
    """接口与 FewShotAdapter 一致(fit_fewshot / predict)。无记忆库。"""

    def __init__(self, model_size="small", device="cuda", train_steps=4000,
                 image_size=256, lr=1e-4, seed=42):
        self.size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_steps = train_steps
        self.image_size = image_size
        self.lr = lr
        self.seed = seed
        get_pdn = get_pdn_small if model_size == "small" else get_pdn_medium
        self.teacher = get_pdn(OUT).eval().to(self.device)
        sd = torch.load(_WEIGHTS / f"teacher_{model_size}.pth", map_location="cpu")
        self.teacher.load_state_dict(sd)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = get_pdn(2 * OUT).to(self.device)
        self.ae = get_autoencoder(OUT).to(self.device)
        self.t_mean = self.t_std = None
        self.q = None                                  # (q_st_a, q_st_b, q_ae_a, q_ae_b)
        self.threshold = None
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    def _prep(self, img):
        """(3,H,W)或(1,3,H,W)[0,1] → (1,3,256,256) ImageNet 归一化。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img.to(self.device), size=self.image_size,
                            mode="bilinear", align_corners=False)
        return (img - self._mean) / self._std

    @torch.no_grad()
    def _teacher_norm(self, normals):
        outs = [self.teacher(self._prep(x)).mean(dim=[0, 2, 3]) for x in normals]
        self.t_mean = torch.stack(outs).mean(0)[None, :, None, None]
        dists = [((self.teacher(self._prep(x)) - self.t_mean) ** 2).mean(dim=[0, 2, 3])
                 for x in normals]
        self.t_std = torch.stack(dists).mean(0)[None, :, None, None].sqrt()

    def fit_fewshot(self, normal_images, defect_images=None):
        torch.manual_seed(self.seed)
        k = max(1, len(normal_images) // 10)
        val_n, train_n = normal_images[:k], normal_images[k:]
        if not train_n:
            train_n = normal_images
        self._teacher_norm(train_n)
        self.student.train(); self.ae.train()
        opt = torch.optim.Adam(itertools.chain(self.student.parameters(), self.ae.parameters()),
                               lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, int(0.95 * self.train_steps), 0.1)
        loader = itertools.cycle(train_n)
        for _ in range(self.train_steps):
            img = _augment(next(loader))            # 几何增广扩充少样本正常流形
            x = self._prep(img)
            x_ae = self._prep(_color_jitter(img))
            with torch.no_grad():
                t_st = (self.teacher(x) - self.t_mean) / self.t_std
            s_st = self.student(x)[:, :OUT]
            d = (t_st - s_st) ** 2
            d_hard = torch.quantile(d, 0.999)
            loss_hard = d[d >= d_hard].mean()
            ae_out = self.ae(x_ae)
            with torch.no_grad():
                t_ae = (self.teacher(x_ae) - self.t_mean) / self.t_std
            s_ae = self.student(x_ae)[:, OUT:]
            loss = loss_hard + ((t_ae - ae_out) ** 2).mean() + ((ae_out - s_ae) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        self.student.eval(); self.ae.eval()
        self._map_norm(val_n if val_n else train_n)
        if defect_images:
            ns = [self._image_score(x)[0] for x in normal_images]
            ds = [self._image_score(x)[0] for x in defect_images]
            self.threshold = FewShotAdapter._calibrate(ns, ds)
        return self.threshold

    @torch.no_grad()
    def _maps(self, img):
        x = self._prep(img)
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = self.student(x)
        a = self.ae(x)
        map_st = ((t - s[:, :OUT]) ** 2).mean(1, keepdim=True)
        map_ae = ((a - s[:, OUT:]) ** 2).mean(1, keepdim=True)
        return map_st, map_ae

    @torch.no_grad()
    def _map_norm(self, val_normals):
        sts, aes = [], []
        for x in val_normals:
            st, ae = self._maps(x)
            sts.append(st); aes.append(ae)
        sts, aes = torch.cat(sts), torch.cat(aes)
        self.q = (torch.quantile(sts, 0.9), torch.quantile(sts, 0.995),
                  torch.quantile(aes, 0.9), torch.quantile(aes, 0.995))

    @torch.no_grad()
    def _image_score(self, img):
        st, ae = self._maps(img)
        if self.q is not None:
            st = 0.1 * (st - self.q[0]) / (self.q[1] - self.q[0])
            ae = 0.1 * (ae - self.q[2]) / (self.q[3] - self.q[2])
        combined = 0.5 * st + 0.5 * ae
        return float(combined.max()), None

    @torch.no_grad()
    def score_large(self, img, max_size=1536):
        """整图全卷积推理(ST 主分支,不分块不降采样)。PDN 全卷积→大图一次前向出全分辨率图。
        只超 max_size 时才等比缩(控显存)。AE 分支固定256不参与大图。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        if max(img.shape[-2:]) > max_size:
            s = max_size / max(img.shape[-2:])
            img = F.interpolate(img, scale_factor=s, mode="bilinear", align_corners=False)
        x = (img - self._mean) / self._std
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = self.student(x)[:, :OUT]
        map_st = ((t - s) ** 2).mean(1)
        return float(map_st.max())

    @torch.no_grad()
    def score_images(self, imgs, batch=16):
        """批量打分(供分块大图批处理):imgs 列表 → 每图异常分(map 最大值)。"""
        out = []
        for i in range(0, len(imgs), batch):
            x = torch.cat([self._prep(im) for im in imgs[i:i + batch]], 0)
            t = (self.teacher(x) - self.t_mean) / self.t_std
            s = self.student(x)
            a = self.ae(x)
            map_st = ((t - s[:, :OUT]) ** 2).mean(1, keepdim=True)
            map_ae = ((a - s[:, OUT:]) ** 2).mean(1, keepdim=True)
            if self.q is not None:
                map_st = 0.1 * (map_st - self.q[0]) / (self.q[1] - self.q[0])
                map_ae = 0.1 * (map_ae - self.q[2]) / (self.q[3] - self.q[2])
            comb = 0.5 * map_st + 0.5 * map_ae
            out.extend(comb.flatten(1).max(1).values.tolist())
        return out

    def predict(self, img):
        t0 = time.perf_counter()
        s, _ = self._image_score(img)
        lat = (time.perf_counter() - t0) * 1000.0
        is_def = bool(self.threshold is not None and s >= self.threshold)
        return {"score": s, "is_defect": is_def, "latency_ms": lat}


def _color_jitter(img):
    """轻量颜色抖动(替代 torchvision ColorJitter,作用于 [0,1] 张量),供 AE 分支增广。"""
    f = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
    return (img * f).clamp(0, 1)


def _augment(img):
    """几何增广(翻转 + 90°旋转):扩充少样本正常流形,降低欠训练误报。img (3,H,W)。"""
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, [-1])
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, [-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        img = torch.rot90(img, k, [-2, -1])
    return img
