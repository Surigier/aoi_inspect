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
                 image_size=256, lr=1e-4, seed=42, compile_infer=False, n_students=1):
        self.size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_steps = train_steps
        self.image_size = image_size
        self.lr = lr
        self.seed = seed
        self.compile_infer = compile_infer            # 推理期 torch.compile 加速(fit后启用)
        self._compiled = False
        # n_students>1:多种子学生集成(仅检测分路径,教师共享;定位/残差路径仍用主学生)。
        # 实测(run_ead_ensemble.py):工作类方差收窄~2×、现场一次fit下限+0.012、均值零代价;
        # fit不计时→训练免费,推理多一次学生前向。
        self.n_students = max(1, int(n_students))
        self._get_pdn = get_pdn_small if model_size == "small" else get_pdn_medium
        self.teacher = self._get_pdn(OUT).eval().to(self.device)
        sd = torch.load(_WEIGHTS / f"teacher_{model_size}.pth", map_location="cpu")
        self.teacher.load_state_dict(sd)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = self._get_pdn(2 * OUT).to(self.device)
        self.ae = get_autoencoder(OUT).to(self.device)
        self.t_mean = self.t_std = None
        self.q = None                                  # (q_st_a, q_st_b, q_ae_a, q_ae_b)
        self.pairs = None                              # [(student, ae, q)](n_students>1时)
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

    def _train_pair(self, train_n, seed):
        """训一个(student, ae)对(教师冻结共享)。返回训好的 eval 态模型对。"""
        torch.manual_seed(seed)
        student = self._get_pdn(2 * OUT).to(self.device)
        ae = get_autoencoder(OUT).to(self.device)
        student.train(); ae.train()
        opt = torch.optim.Adam(itertools.chain(student.parameters(), ae.parameters()),
                               lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, int(0.95 * self.train_steps), 0.1)
        loader = itertools.cycle(train_n)
        for _ in range(self.train_steps):
            img = _augment(next(loader))            # 几何增广扩充少样本正常流形
            x = self._prep(img)
            x_ae = self._prep(_color_jitter(img))
            with torch.no_grad():
                t_st = (self.teacher(x) - self.t_mean) / self.t_std
            s_st = student(x)[:, :OUT]
            d = (t_st - s_st) ** 2
            d_hard = torch.quantile(d, 0.999)
            loss_hard = d[d >= d_hard].mean()
            ae_out = ae(x_ae)
            with torch.no_grad():
                t_ae = (self.teacher(x_ae) - self.t_mean) / self.t_std
            s_ae = student(x_ae)[:, OUT:]
            loss = loss_hard + ((t_ae - ae_out) ** 2).mean() + ((ae_out - s_ae) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        student.eval(); ae.eval()
        return student, ae

    def fit_fewshot(self, normal_images, defect_images=None):
        torch.manual_seed(self.seed)
        k = max(1, len(normal_images) // 10)
        val_n, train_n = normal_images[:k], normal_images[k:]
        if not train_n:
            train_n = normal_images
        self._teacher_norm(train_n)
        self.pairs = []
        for i in range(self.n_students):
            student, ae = self._train_pair(train_n, self.seed + i)
            q = self._map_norm(val_n if val_n else train_n, student, ae)
            self.pairs.append((student, ae, q))
        self.student, self.ae, self.q = self.pairs[0]         # 主学生(定位/残差路径用)
        self._maybe_compile(train_n[0] if train_n else None)
        if defect_images:
            ns = [self._image_score(x)[0] for x in normal_images]
            ds = [self._image_score(x)[0] for x in defect_images]
            self.threshold = FewShotAdapter._calibrate(ns, ds)
        return self.threshold

    def _maybe_compile(self, warm_img):
        """fit 后用 torch.compile 加速推理(dynamic 应对变尺寸大图)。实测方形-24%、细长-56%。
        warm_img 用于预热(编译在不计时的 fit 阶段完成);失败则静默回退 eager。"""
        if not self.compile_infer or self._compiled or self.device == "cpu":
            return
        try:
            self.teacher = torch.compile(self.teacher, dynamic=True)
            if self.pairs:
                self.pairs = [(torch.compile(s, dynamic=True), a, q) for s, a, q in self.pairs]
                self.student = self.pairs[0][0]
            else:
                self.student = torch.compile(self.student, dynamic=True)
            self._compiled = True
            if warm_img is not None:
                self.score_large(warm_img)                # 触发编译(untimed)
        except Exception:
            self._compiled = False                        # 回退 eager,不影响正确性

    @torch.no_grad()
    def _maps(self, img, student=None, ae_model=None):
        student = student if student is not None else self.student
        ae_model = ae_model if ae_model is not None else self.ae
        x = self._prep(img)
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = student(x)
        a = ae_model(x)
        map_st = ((t - s[:, :OUT]) ** 2).mean(1, keepdim=True)
        map_ae = ((a - s[:, OUT:]) ** 2).mean(1, keepdim=True)
        return map_st, map_ae

    @torch.no_grad()
    def _map_norm(self, val_normals, student=None, ae_model=None):
        sts, aes = [], []
        for x in val_normals:
            st, ae = self._maps(x, student, ae_model)
            sts.append(st); aes.append(ae)
        sts, aes = torch.cat(sts), torch.cat(aes)
        q = (torch.quantile(sts, 0.9), torch.quantile(sts, 0.995),
             torch.quantile(aes, 0.9), torch.quantile(aes, 0.995))
        if student is None:
            self.q = q
        return q

    @torch.no_grad()
    def _image_score(self, img):
        """检测分。n_students>1:逐学生打分取平均(与run_ead_ensemble探针口径一致)。"""
        pairs = self.pairs if self.pairs else [(self.student, self.ae, self.q)]
        scores = []
        for student, ae_model, q in pairs:
            st, ae = self._maps(img, student, ae_model)
            if q is not None:
                st = 0.1 * (st - q[0]) / (q[1] - q[0])
                ae = 0.1 * (ae - q[2]) / (q[3] - q[2])
            scores.append(float((0.5 * st + 0.5 * ae).max()))
        return sum(scores) / len(scores), None

    @torch.no_grad()
    def score_large(self, img, max_size=1280, max_pixels=None, use_half=True):
        """整图全卷积推理(ST 主分支,不分块不降采样)。PDN 全卷积→大图一次前向出全分辨率图。
        等比缩以同时满足:长边≤max_size、面积≤max_pixels(后者保证方图也达标延时,
        因延时∝面积;只卡长边时方图面积可达扁图3倍→延时爆)。
        use_half:FP16 autocast,卷积吞吐~翻倍→全分辨率(1280)也<200ms@2060,免降分辨率砸精度。
        AE 分支固定256不参与大图。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        h, w = img.shape[-2:]
        s = 1.0
        if max(h, w) > max_size:
            s = min(s, max_size / max(h, w))
        if max_pixels is not None and h * w * s * s > max_pixels:
            s = min(s, (max_pixels / (h * w)) ** 0.5)
        if s < 1.0:
            img = F.interpolate(img, scale_factor=s, mode="bilinear", align_corners=False)
        x = (img - self._mean) / self._std
        half = use_half and self.device != "cpu"        # FP16 仅 GPU;CPU 走 FP32(OpenVINO另导)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half):
            t = (self.teacher(x) - self.t_mean) / self.t_std   # 教师前向一次,学生集成共享
            students = [p[0] for p in self.pairs] if self.pairs else [self.student]
            scores = []
            for student in students:
                st = student(x)[:, :OUT]
                scores.append(float(((t - st) ** 2).mean(1).max()))
        return sum(scores) / len(scores)

    @torch.no_grad()
    def anomaly_map_large(self, img, max_size=1152, max_pixels=1_400_000,
                          use_half=True, out_hw=None):
        """整图全卷积像素级异常图(ST 主分支),上采样到 out_hw(默认原图 HxW)。
        用于像素级分割/定位评分(赛题按定位评准确率)。返回 (H,W) numpy。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        H, W = img.shape[-2:]
        h, w = H, W
        s = 1.0
        if max(h, w) > max_size:
            s = min(s, max_size / max(h, w))
        if max_pixels is not None and h * w * s * s > max_pixels:
            s = min(s, (max_pixels / (h * w)) ** 0.5)
        x_img = F.interpolate(img, scale_factor=s, mode="bilinear", align_corners=False) if s < 1.0 else img
        x = (x_img - self._mean) / self._std
        half = use_half and self.device != "cpu"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half):
            t = (self.teacher(x) - self.t_mean) / self.t_std
            st = self.student(x)[:, :OUT]
            map_st = ((t - st) ** 2).mean(1, keepdim=True)        # (1,1,h',w')
        oh, ow = out_hw if out_hw is not None else (H, W)
        amap = F.interpolate(map_st.float(), size=(oh, ow), mode="bilinear", align_corners=False)
        return amap[0, 0].cpu().numpy()

    @torch.no_grad()
    def residual_map_large(self, img, max_size=1152, max_pixels=1_400_000, use_half=True):
        """整图 per-pixel 多通道残差 (C=OUT, h', w'):teacher-student 逐通道平方差(未mean)。
        供监督分割头(用30张标注缺陷掩膜训1×1 conv)——比单通道异常图信息更丰富。
        返回 (C,h',w') float tensor(device 上),及相对原图的缩放比 s。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        h, w = img.shape[-2:]
        s = 1.0
        if max(h, w) > max_size:
            s = min(s, max_size / max(h, w))
        if max_pixels is not None and h * w * s * s > max_pixels:
            s = min(s, (max_pixels / (h * w)) ** 0.5)
        x_img = F.interpolate(img, scale_factor=s, mode="bilinear", align_corners=False) if s < 1.0 else img
        x = (x_img - self._mean) / self._std
        half = use_half and self.device != "cpu"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half):
            t = (self.teacher(x) - self.t_mean) / self.t_std
            st = self.student(x)[:, :OUT]
            res = (t - st) ** 2
        return res[0].float()                          # (OUT, h', w')

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
