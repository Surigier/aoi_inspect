# rddn_yolo/ —— 参考图条件YOLO候选框提议器(独立实验子工程)

## 定位与隔离原则

这是一个**独立的实验子工程**,和`aoi/`(生产管线,`CompetitionLargeDetector`)物理隔离:

- **不修改`aoi/`下任何现有文件**。只允许只读import其中的通用工具(如`aoi.imageio.load_fast`),
  不允许改动它们的代码或行为。
- 生产管线(`aoi/competition.py`的`CompetitionLargeDetector`)在这个子工程完成前、
  完成后、判负后,行为都完全不变,随时可以`git log`回到任意commit干净复现原来的结果。
- 只有当这里的机制经过真实数据OOF验证证明净正,才会在`aoi/competition.py`里加一个
  **新的、默认关闭的可选阶段**去调用它(参照今天SAM门控/crop_cascade/component_graph/
  boundary_refine一致的"opt-in,零回退"接入方式)——接入本身也是纯增量,不改写现有逻辑。

## 架构(用户2026-07-20提议,已讨论确认)

```
EAD+DINO判异常(生产现有,不变)
   ↓
切片RGB + 配准差异通道 → 小型YOLO候选框(本工程新增)
   ↓
框内WRN热图/现有分割头精修(复用现有WRN特征提取器,可能需独立训练的crop-head)
   ↓
低置信时回退原始WRN全图结果(零回退安全网)
```

核心与crop_cascade(已判负,-0.059)的区别:crop_cascade的候选生成器是无监督阈值
直接OR进结果,配准/光照噪声直接变假阳性;这里差异通道只是喂给**监督**YOLO的输入
特征,真假由30张真实缺陷+海量normal-normal负样本学出来,不是拍脑袋阈值。

## 进度

1. ✅ `diff_channels.py`:ECC对齐+差异通道构造(Lab色差/灰度梯度差/局部SSIM残差)
2. ✅ `diag_diff_signal.py`:可行性探针,Real-IAD 5个电子/手机件类(phone_battery/
   pcb/sim_card_set/usb/switch)真实数据结果——mask/背景差异强度比值中位5.28(全部
   远高于噪声水平1.0),候选框召回@IoU0.3均值0.430(裸阈值零学习,switch达0.640)。
   信号存在且比crop_cascade当初的处境更扎实,支持继续投入。
3. ⬜ 数据集导出:GT掩膜→YOLO格式框标注,跨12个Real-IAD电子件类目(~82000张图,
   本机`data/_dl/Real-IAD/`)组装训练/验证集
4. ⬜ 6通道权重手术:ultralytics YOLOv8n第一层从3通道扩到6通道,RGB部分权重保留,
   新增3通道零/均值初始化
5. ⬜ 训练循环(跨类预训练,学"什么残差是真缺陷/什么是噪声"的通用先验)
6. ⬜ 独立crop-head(框内精修,不能复用全图头统计量——roi_zoom失败的教训)
7. ⬜ OOF门控(k折,不用单次留出——今天反复验证过单次小样本门控不可靠)
8. ⬜ 延时实测(新增一整个检测器前向,不能假设"轻量就没事")
9. ⬜ 真实数据A/B(LOCO/AD2/Real-IAD多类),净正才考虑接入`aoi/competition.py`

## 本机数据

`data/_dl/Real-IAD/`下12个类目有真实图像(json标注在`data/_dl/realiad_jsons/`):
phone_battery(6550张)/pcb(7299)/sim_card_set(7099)/usb(6804)/usb_adaptor(6535)/
switch(7044)/button_battery(7150)/terminalblock(7058)/transistor1(7117)/
regulator(6180)/audiojack(6346)/end_cap(7193)。json格式与`scripts/run_scorecard.py`
的`prep_realiad()`一致(`train`/`test`列表,`category`/`anomaly_class`/`image_path`/
`mask_path`字段)。
