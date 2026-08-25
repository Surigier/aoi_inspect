1. 全部用中文回复
2. 称呼规则：每次回复必须使用 leon 作为称呼
3. 遇到不确定的代码设计问题时，必须先询问 leon，不得直接行动
4. 代码兼容性：不能写兼容性代码，除非主动提要求
5. 代码风格：必须遵守 karpathy-guidelines 里的四条行为准则(Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution)
6.不要总是自大，总是不看别人的代码，不要总是自己猜


---

# 【接手须知】新会话先读这段(2026-08-24 更新)

赛题:2026中国研究生AI大赛·华为赛题一《可自学习的AOI实时在线AI质检》,**8月底截止**。
少样本工业异常检测+定位:100正常+30缺陷现场迁移(**fit不计时**),再对1000+测试图
检测+定位,**<200ms@2060**,2500²输入,隐藏域=手机部件(屏/电池/中框)。

## 当前成绩(真口径,4060L)

| 指标 | 值 |
|---|---|
| 图级acc / 含漏检IoU / 框命中@0.5 | **0.920 / 0.511 / 0.582** |
| 缺陷类型归属(端到端,预测掩膜) | **88%**(启发式基线40%) |
| 延时 @2500²真尺寸 | **中位97ms / p90 99ms**(预算200ms) |
| 延时 @小图缺陷图 | p90 159ms |
| 单元测试 | 62 passed |

## 生产架构一句话

`locate()` = EfficientAD(2学生)+DINOv2双判据检测 → 判正常立即早退 → WRN浅层(1,2)@512
+双头联合训练监督分割 + SAM受控精化 → VLM蒸馏的类型头。全部细节见下文各条目。

## 下一步(按性价比排序)

1. **Real-IAD全类目成绩单** —— `data/_dl/Real-IAD` 有12类目每类~2400张OK图+真实像素
   掩膜,现有成绩单**只用了2个(pcb/phone_battery)**。`prep_realiad(cat)`全参数化,
   另10个直接能跑(sim_card_set/usb/usb_adaptor/audiojack/button_battery/switch/
   terminalblock/transistor1/regulator/end_cap)。约20分钟/类目。这是手机部件域真实
   成绩单的正路,**别去补MSD那20张正常图**。
2. **重写 docs/delivery/结果报告.md** —— 全篇AUROC,直接违反铁律#1。且算分是
   方案完整度50%+答案准确率20%+检测时间30%,**准确率只占竞赛得分20%、总分12%**,
   文档影响的那50%才是大头。(可解释性文档.md/使用说明.md已按真架构重写完)
3. **2060真机延时 + 持续满负载1000+张的热漂移** —— 检测时间占30%,比准确率还重。
4. **油污归类口径待定**(需leon拍板或出题方答复):提示词那条"材料没破损就判色彩变化"
   是在MVTec上调的,搬到手机域后把污渍/油污都推向"色彩变化",而工业AOI习惯上脏污
   一般算"外观缺陷"。**一行提示词的事,不是架构问题。**

## 三个最容易重复踩的坑

- **GPU脏卡陷阱**:不同热状态/负载下测的延时不可比。必须同一段GPU状态下背靠背A/B,
  且预热后再计时。曾把906ms当回归上报,实为`compile_infer=False`+未预热+全缺陷图。
- **只测缺陷图不测正常图**:GCAD-EmbedAE因此被误接入生产,图级acc从0.902崩到0.703。
  **任何改动都必须用含正常图的完整成绩单验证。**
- **记忆库自匹配**:辅助分支拿正常图建库再给同一批正常图打分→距离0→统计量无意义。
  已用OOF留出修掉(类型归属0%→50%)。

---

## Plugins

This project uses the following Claude Code plugins (installed at user scope):

- **superpowers@superpowers-dev** — Full software development workflow with brainstorming, TDD, plan-driven execution, and subagent-driven development skills. Activate via the Skill tool.
- **andrej-karpathy-skills@karpathy-dev** — Behavioral guidelines (karpathy-guidelines skill) to reduce common LLM coding mistakes: Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution.

## Karpathy Coding Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

# aoi_inspect — 项目状态与工作准则(跨机器/跨会话迁移用)

2026研究生AI大赛·华为赛题一《可自学习的AOI实时在线AI质检》,截止2026年8月。
few-shot工业异常检测+定位:每产品100正常+30标注缺陷现场迁移(fit不计时),
测试图判缺陷+定位,<200ms@2060(2500²输入,隐藏域=手机部件:屏幕/电池/中框)。

## 铁律(用户明确定下,任何会话都必须遵守)

1. **只用评分真口径汇报**:图级acc/逐图含漏检IoU/框命中@0.5/延时。严禁拿AUROC说事。
2. **定位精度用户没认可前,不做答辩材料、不自作主张转方向**。
3. **零回退纪律**:任何新机制必须真实数据留出/OOF验证净正才进生产;验负就回退并留档
   (commit message+代码注释),不带病上线。
4. AD2数据集只用于大图延时/形状压测,不做精度叙事;不要碰"反光表面"话题。
5. 简洁直接,少问多做,快速定位问题;负结果照实说,不粉饰。

## 当前生产架构(competition.py CompetitionLargeDetector)

- 检测:EfficientAD学生-教师(多种子ead_students=2,无记忆库延时恒定)+ DINOv2受控图级门
- 定位:WRN50浅层(1,2)@512监督分割头 = **旧实现`_seg_head_old_ae5fbbb.py`**(双头集成+
  pooled-F1自洽阈值)——见下"seg_head回退"
- 模板差分特征(tmpl_ref.py金模板+ECC):fit留出自动选,pcb类刚性件+21%
- SAM边界精化:逐区域OOF门控(sam_refine.py),5域实测全判reject_all(短路省40-60ms)
- DINOv2图级co-detector:**默认永远融合**(不再由fit侧3折CV决定开关,commit 4cdc115)
- 组件图(component_graph.py):逻辑缺陷保险丝,严格门控,只在"seg连fit图都拟合不了"
  的极端逻辑缺陷产品上启用;"默认开"已被否(纹理类-0.218灾难)
- 辅助分支(色彩/尺寸/结构):只做缺陷类型归属,不参与检测
- 热路径:正常图EAD/DINO判定后立即返回;单次GPU上传;submit.py目录评测双缓冲
- 延时探针:真实原生文件路径(det.probe_paths),预算190ms自适应裁剪

## 用户反馈驱动优化 + 缺陷类型归属修复(2026-08-21)

赛题原文该条完整要求:"当系统**误检或漏检**时,操作员可提供**实时**反馈,系统应能
**回溯检测逻辑**,动态调整模型参数"。此前只做到"漏检反馈能提升定位精度"一条,补齐:

- **①【实时性】误检和漏检**两条反馈路径**都**跳过EAD学生重训(`retrain_ead`开关,competition.py/
  tiled_efficientad.py/active_learning.py三处透传)。依据:`TiledEfficientAD.
  fit_fewshot`里学生只在`norm_tiles`上训(缺陷图传的是None),**缺陷图只参与阈值
  标定**——操作员标记漏检时重训学生纯属浪费。`ActiveLearningLoop.feedback()`自动
  判断:`is_defect=True`(只新增缺陷图)走增量,`is_defect=False`(新增正常图,学生
  必须吃到)走完整fit。**实测hazelnut:完整fit 1072s → 增量fit 105s,加速10.2倍**,
  "实时反馈"从完全不成立变成可用。
  - **误检路径也走快路径,是实测定的不是想当然**(scripts/run_fp_margin.py):直觉
    是"新增正常图→学生必须重训才能吃到",但三个类目的留出正常图实测**零假阳性**
    (阈值标定偏保守),没有天然误检样本可测,于是改测**安全边距**这个连续量
    (margin=判定阈值-融合分,越大越不易被误判):把留出正常图里融合分最高
    (最接近被误判)的3张标记为误检,hazelnut结果——**快路径边距+0.326、完整路径
    +0.269,快路径反而更好**。机制解释:修复误检靠阈值/DINO门/像素阈值重标,都不经过
    学生权重;重训学生反而让学生把这张新正常图学好→它EAD分下降→阈值标定时它不再是
    "高分正常样本",阈值少上移一点。**泛化性**:未被反馈的其余正常图边距只+0.021,
    说明反馈只修被标记的那张及近邻,不会几张反馈就把整体灵敏度调没(期望行为)。
    学生权重陈旧性由离线完整重训兜底(fit_fewshot默认retrain_ead=True),不占用
    操作员实时路径。
  - ⚠️注意该实验用train_steps=100(为快速造场景),学生训练占比小,所以快/完整
    路径耗时只差1.2倍(493s vs 590s);**生产train_steps=10000下省掉的才是17分钟
    的大头**,真实加速比以phone_battery生产配置实测的1193s→251s为准。
- **②【回溯检测逻辑】`det.explain(img)`**(competition.py):摊开完整判定链路——
  各分支原始分/z分、生效阈值、**谁主导判定**(EAD还是DINO)、灰区/GCAD救援门是否
  触发、类型归属的分支竞争过程、掩膜经过哪些精化模块、**延时自适应裁掉了什么**。
  **不碰热路径**:`locate()`一行未改,explain是冷路径单独重算,不占200ms预算。
- **③【缺陷类型归属】查出并修掉一个"统计量自己匹配自己"的真bug**(explain()第一次
  跑就暴露的):三个辅助分支(色彩/尺寸/结构)都是记忆库结构——`fit(normals)`把正常
  图特征存进bank,`infer`算到bank的最近邻距离。原写法**bank用normals建、又拿同一批
  normals打分**,正常图匹配到自己、距离恒为0:色彩分支实测mean=0/std=0(2维色度
  数值干净,精确为0),结构分支实测mean=1.97/std=0.35——**那不是真实分布,是
  torch.cdist用矩阵乘法算欧氏距离的数值误差**(1536维深度特征数值大,误差显著)。
  拿假统计量做z归一,跨分支完全不可比:结构分支z爆到197、其他只有1~7,
  **类型归属100%误判**(hazelnut的crack/cut/hole + pill的color共36张,全判成
  "缺件/逻辑",一张没对)。**修法**:`_oof_aux_normal_scores()`用2折留出算统计量
  (A半建库打B半、B半建库打A半),统计量算完再用全量normals重建生产库。
  **修复后类型准确率 0% → 50%**(hazelnut 13/18=72%、pill 6/17、carpet 4/12、
  metal_nut 7/13)。**影响范围收敛**:`self.stats`只被`_ztype`和`explain`使用,
  **检测/定位完全不碰**,acc/IoU/框命中不受影响,无需重跑成绩单。
  - **残留50%误判是另一个根因,未修**:分支本身的判别力不够,不是归一化问题。
    实测证据:pill的真实色彩缺陷里,色彩分支的z分有7/17次只排第3——`ColorADBranch`
    把图下采样到320²后按16×16格取**每格色度均值**(一格约20px),pill的小色斑被
    均值平摊掉了。试过的补救:①"把EAD降级成兜底、只在3个专家分支里选"——**不成立**,
    hazelnut会永远判不出"外观缺陷"(0/18);②色彩分支格子加密(grid 16→32),
    色彩类合计40%→45%,**边际提升,未采纳**(只在3个色彩类目上测过,对其他类目的
    副作用未验证)。要再往上需要重做分支判别力,工程量大且**类型归属不在赛题计分
    公式内**(竞赛得分=方案完整度+答案准确率+检测时间),暂停于此并如实记录。

## 赛题完整度核查(2026-07-24,重读赛题原文后发现的重大方向调整)

赛题原文竞赛得分(60%)= **方案完整度(50%)+ 答案准确率(20%)+ 检测时间(30%)**,另有
专家评分(40%)看提交的可解释性文档+使用说明。**此前一个月+今天几乎全部精力都在打磨
"答案准确率"这20%权重的定位IoU**(WRN-LoRA/Top-1 ROI/UniVAD/DCP-SFR/TTA等一串实验),
权重最大的"方案完整度"(50%)和专家评分要看的交付文档反而被忽视。核查结果:

- `docs/delivery/可解释性文档.md`/`使用说明.md`**严重过时**(最后改动2026-07-07,
  `aoi/competition.py`最后改动2026-07-22,不算今天工作)。文档描述的是**另一套架构**
  (PatchCore记忆库5分支融合+ActiveLearningLoop+LLM报告,对应`submit.py`的
  `_run_small`/`_run_zeroshot`小图/零样本路径),而真正跑2500²真实测试图的
  `_run_large`→`CompetitionLargeDetector`(EAD+DINO+WRN+SAM+crop_cascade+
  component_graph)在文档里只字未提。**待办:文档要等下面几项功能补完后统一重写**
  (用户明确要求:先把题目要求的功能做完,再一次性写文档,不要边做边改)。
- `submit.py`本身路由是对的(长边≥1024→`_run_large`用`CompetitionLargeDetector`,
  否则`_run_small`;`--zeroshot`走独立CLIP路径),不是要推倒重来,只是文档没跟上代码。
- **用户反馈驱动优化("赛题必做支柱")已验证接入`CompetitionLargeDetector`可行且有效**
  (`aoi/active_learning.py`扩展了`defect_masks`可选追踪,向后兼容,见
  `scripts/run_active_learning_large.py`):phone_battery真实验证,初始仅10张缺陷
  fit时留出test IoU=0.236/框命中=0.267,模拟操作员反馈5张漏检(defect集10→15张,
  每次反馈重跑完整`fit_fewshot`)后同一留出集IoU=0.335(+0.099)/框命中=0.333
  (+0.067)——反馈机制在生产大图架构上真实生效,不是理论上能接而已。
- **零样本冷启动路径评估为"设计上合理,不需要改"**:`submit.py --zeroshot`走独立
  `ZeroShotAdapter`(CLIP文本提示),不复用`CompetitionLargeDetector`——这是必然的,
  因为EAD/WRN/SAM/DINO所有机制都需要至少若干张校准图,零样本(0张任何样本)只有
  纯CLIP这类完全免标定的方法能做。文档需要写清楚这个架构决策,不是缺口。
- **CPU<2s挑战目标:当前重架构不可行,需诚实记录**。`scripts/run_cpu_latency.py`
  测的是旧的OpenVINO ResNet18分块架构(和`CompetitionLargeDetector`完全无关,已过时)。
  真实测试(`scripts/run_cpu_latency_large.py`,device=cpu,仅8张正常+3张缺陷、
  不训掩膜的最小fit_fewshot)跑了**近2小时仍未跑完fit阶段**(CPU占用持续
  1100%+,确认是真实计算量大而非卡死,非死锁)。**结论:现有EAD双学生+WRN+
  DINOv2+SAM的重组合在CPU上不具备实用性,这是"可挑战目标"而非硬性门槛,文档里
  如实说明"未达成"即可,不必现在勉强凑数字**。模型体量本身不算大(WRN浅层
  4.1M+EAD PDN×3约9.2M+DINOv2 ViT-S/14约22M+MobileSAM约10M,合计约45M参数,
  "鼓励使用小模型"这条不算严重违背),瓶颈是CPU上多模型级联的计算延时而非参数量。
- **视频路径在当前生产架构上验证正常(修正了一次虚惊)**:`scripts/verify_video_gate.py`
  (2026-07-07旧脚本)对cable/missing_cable取held-out缺陷段做`frame_score`+滑动平均+
  事件聚合验证,首次跑出"新-融合门缺陷检出=❌"看似视频路径回归——排查后发现是脚本本身
  的bug:该类实际只有12张缺陷图,脚本硬编码`dfiles[15:23]`(假设>=23张)在12张的列表
  上越界,Python越界切片静默返回空列表而不报错,导致"缺陷段"其实是空,测的是一段全
  正常帧。修复切片(改为按实际数量动态取最后4张做held-out)后重跑:新-融合门正确识别
  事件(12,13)精准覆盖held-out缺陷帧[10,11,12,13],误报=无,平滑分数在缺陷帧内清晰起峰
  (3.59→9.16)后回落;旧-EAD门(纯EAD,无DINO融合,仅作对比基线)则触发覆盖全片的
  单一误报事件——证实DINO融合修复的价值同样适用于视频路径,不只是静态图。结论:视频
  能力是完整的,此前"完整度审查"没有发现真实回归,是一次被脚本bug放大的虚惊。

## 当前成绩(2026-08-17最新,seg_head双头联合训练已上生产,本机4060L)

| 类            | 图级acc   | 含漏检IoU | 纯定位IoU | 框命中@0.5 | locate |
| ------------- | --------- | --------- | --------- | ---------- | ------ |
| hazelnut      | 0.921     | 0.598     | 0.647     | 0.639      | 180ms  |
| cable         | **0.927** | **0.802** | 0.802     | **0.933**  | 155ms  |
| pill          | 1.000     | 0.456     | 0.456     | 0.475      | 148ms  |
| pcb           | 0.825     | 0.300     | 0.309     | 0.312      | 84ms   |
| phone_battery | 0.925     | 0.399     | 0.410     | 0.550      | 93ms   |
| **均值**      | **0.920** | **0.511** | 0.525     | **0.582**  |        |

**vs 上一版基线(0.909/0.506/0.527/0.554)**:图级acc **+0.010**、含漏检IoU
**+0.005**、框命中 **+0.028**(赛题评分口径的三项全部改善);纯定位IoU -0.002
(该指标不在赛题评分口径内,赛题看含漏检IoU)。唯一变更是seg_head双头联合训练
(见下"双头联合训练"条目),零延时代价、不碰骨干、推理结构不变。

上一版基线(2026-07-20,双头各自独立训):

| 类 | 图级acc | 含漏检IoU | 纯定位IoU | 框命中@0.5 |
| --- | --- | --- | --- | --- |
| hazelnut | 0.882 | 0.627 | 0.717 | 0.611 |
| cable | 0.927 | 0.811 | 0.811 | 0.933 |
| pill | 1.000 | 0.440 | 0.440 | 0.426 |
| pcb | 0.825 | 0.253 | 0.260 | 0.287 |
| phone_battery | 0.912 | 0.401 | 0.407 | 0.512 |
| **均值** | **0.909** | **0.506** | 0.527 | **0.554** |

更早历史基线:含漏检IoU均值0.484/框0.600。cable问题**已彻底根治并确定性复现**
(见下),不再是开放问题。
延时:最轻档p90≈205ms@4060L(compile_infer=True,悲观代理);2060真机待验。

**缺陷类型分层成绩单补充(2026-07-24,scripts/run_scorecard_defect_types.py)**:上表按
产品类目报数,能对应赛题5类缺陷里的3类(常见外观缺陷→hazelnut/缺件少件→cable/
色彩变化→pill),另2类(尺寸偏差/逻辑错误)此前从未用真口径验证过。用MVTec LOCO
真实数据补齐:
| 缺陷类型(LOCO子集) | 图级acc | 含漏检IoU | 纯定位IoU | 框命中@0.5 |
|---|---|---|---|---|
| 常见外观(breakfast_box结构) | 0.583 | 0.151 | 0.189 | 0.240 |
| 常见外观(juice_bottle结构) | 0.815 | 0.221 | 0.293 | 0.228 |
| 缺件/错位/组合(breakfast_box逻辑) | 0.722 | 0.348 | 0.381 | **0.517** |
| 缺件/错位/组合(splicing_connectors逻辑) | 0.714 | 0.143 | 0.180 | 0.114 |
| 缺件/错位/组合(screw_bag逻辑) | 0.685 | 0.095 | 0.130 | 0.052 |
| **均值** | 0.704 | 0.192 | 0.235 | 0.230 |
比现有5类均值(acc=0.902/IoU=0.501)明显更弱,大致和目前最弱的pcb/phone_battery
同量级——这是真实、未经针对性调优的读数,不是坏消息掩饰,恰好说明"逻辑异常"类
仍是薄弱环节,和"pcb/battery微小缺陷是主拉分点"的既有结论方向一致。**重要澄清**:
LOCO的logical_anomalies官方定义是"计数/位置/搭配错误"(比如某格子缺螺丝、线缆长度
配错接头),不完全等同赛题原文"逻辑错误(如顺序错误)"字面意思——"顺序错误"可能指
2500²图像4块1024²拼接的空间顺序,或产线装配的时序先后,这条已列入给出题人的邮件
问题清单,回复前不确定我们现在的机制(component_graph等)是否真的对上号。

**严格按赛题5类缺陷的完整成绩单(2026-07-27,scripts/run_scorecard_5types.py)**:在上面
基础上补齐色彩变化(此前只有pill一类)和缺件少件(此前只有cable一类)的更多真实数据:

| 缺陷类型 | 数据来源 | 图级acc | 含漏检IoU | 框命中@0.5 |
|---|---|---|---|---|
| 常见外观缺陷 | hazelnut | 0.921 | 0.660 | 0.653 |
| 缺件少件 | cable | 0.927 | 0.811 | 0.933 |
| 缺件少件(新) | pcb的QS子集(Real-IAD官方缺陷编码,104张真实缺件标注) | 0.713 | 0.321 | 0.438 |
| 色彩变化 | pill | 1.000 | 0.440 | 0.426 |
| 色彩变化(新) | carpet/leather/metal_nut/wood均值(MVTec官方color子目录) | 0.977 | 0.574 | 0.698 |
| 逻辑错误(如顺序错误) | LOCO 3类均值(定义存疑,见上) | 0.704 | 0.192 | 0.230 |
| **尺寸偏差** | — | **无数据** | **无数据** | **无数据** |

**尺寸偏差彻查结论**:本地7个数据集(MVTec AD/LOCO/Real-IAD/DAGM/VisA/MPDD/pku_pcb)
逐一查过缺陷子类目录名/官方编码表,没有一个明确标注"尺寸偏差"这个缺陷类型
(Real-IAD官方8类编码AK=凹坑/BX=变形/CH=磨损/HS=划伤/PS=破损/QS=缺件/YW=异物/
ZW=污染,没有"尺寸"这一类)。外部搜索找到"Open-Industry Benchmark"(convex/
concave/deformation)但是3D点云数据,和本系统2D图像架构不兼容,且无公开下载链接。
**这不是没做完,是这道题本身缺这块公开语料**,现场只能靠"尺寸"辅助分支(仅类型
归属,未做检测验证)兜底,文档需如实写明此项无数据支撑。

**顺带发现**:pcb-QS(缺件)比pcb整体(acc=0.825/IoU=0.253/框命中=0.287)acc更低
(0.713)但IoU/框命中更高(0.321/0.438)——缺件类一旦被判定为缺陷,定位反而更准
(缺件造成的空洞边界清晰),但检测confidence不如pcb其他缺陷类型(异物/污染/划伤)
稳定,这是个新观察,暂未深究原因。

**端到端2500²联合验证(2026-07-20,run_e2e_2500_check.py)**:此前延时/精度分开验证
(延时用AD2/PKU探针,精度用MVTec/RealIAD/LOCO),首次用当天最终代码在真实大图
(AD2 sheet_metal 4224×1056/vial 1400×1900,真实per-pixel掩膜)联合确认,真实
probe_paths(非重建张量)。结果健康:sheet_metal无需裁剪(lat_trimmed=[],探针146ms)
locate均值95ms/p90=110ms,acc=0.750 IoU=0.414 框=0.271;vial触发裁剪(弃学生2+
max_pixels降900k,探针168ms)但DINO/SAM全程保留(验证了裁剪顺序重排的真实效果)
locate均值122ms/p90=134ms,acc=0.750 IoU=0.622 框=0.583。辅助分支类型归属+视频
frame_score路径也冒烟确认未受今天predict()/locate()重构影响。

**cable问题根治记录(2026-07-20,已解决)**:表面看是"随机性",实为两处独立机制打架。
①精度侧`_calibrate_dino_gate`原用3折CV决定DINO门开关,在cable上是掷硬币(fit侧对
"训练台架↔测试台架"系统性差异结构性看不见)→改为默认永远融合(commit 4cdc115)。
②即便①修好,`_calibrate_latency`延时自适应裁剪梯队仍会为压时间独立砍掉DINO——
旧顺序按"SAM最值钱"的过时认知排的,DINO排第二个被砍,与①的判断完全无关→重排为
student2→max_pixels→SAM→**DINO最后才砍**(commit a6bd551,依据:SAM已证5/5域
reject_all无正贡献,max_pixels对纯定位IoU零影响,DINO是cable唯一救命机制)。
两处都改完后,cable确定性复现acc=0.927(15/15缺陷全命中,40张正常仅4张误报)——
不再随执行顺序/GPU状态波动。**教训:门控/裁剪类改动必须检查是否存在下游独立机制
能把上游判断悄悄推翻,单点验证不代表端到端生效。**

**⚠️新发现的潜在风险(2026-07-25,global_context实验中意外撞见)**:`_calibrate_latency`
硬线超时时仍会把已标定好的DINO门砍掉(见上①②修复里的"DINO排最后才砍"——这条路径
本身没错,但"最后才砍"不等于"不会砍")。今天本机GPU连轴跑了一整天(WRN-LoRA/Top-1
ROI/反馈闭环等多个后台任务,温度一度到87°C)后,同一份代码在9次fit里有5次真的触发
了硬线超时→`lat_trimmed`带上"dino_gate"→cable唯一救命机制被砍。**这不是今天验证
实验的bug,是生产`_calibrate_latency`本身在机器负载/温度较高时会真实发生的行为**——
如果评委机器评测时也处于高负载/高温状态(比如同时跑多个选手提交,或GPU刚跑完别的
任务),cable这类靠DINO门救回来的类目有可能在评委机器上重新暴露旧问题。**待办:
需要专门验证`_calibrate_latency`的硬线阈值(190ms预算)在GPU负载/温度不同状态下
的探针读数稳定性,必要时给"DINO门"设一条比其他裁剪项更高的免砍优先级(比如探针
读数异常高时优先怀疑机器状态而不是砍最后一道保险),而不是依赖"顺序排最后"这种
相对保护。**

## 重大负结果(勿重蹈,证据在commit与代码注释)

- DINO门"3折CV决定开关"已废弃(commit 4cdc115):cable上被证明是掷硬币而非真信号,
  fit侧对"test集系统性漂移"结构性看不见,同seg_head/component_graph今天的教训一致。
  改为默认永远融合(风险不对称:pcb过度触发代价仅-0.011,漏融合代价可达-0.6+)。
- 延时裁剪梯队旧顺序(先砍DINO再砍SAM/max_pixels)已废弃(commit a6bd551):按"SAM
  最值钱"的过时认知排的,与今天SAM 5/5域reject_all、max_pixels零精度影响的新证据
  矛盾,且会悄悄推翻精度侧"永远融合"的决策。重排为student2→max_pixels→SAM→DINO
  (最后才砍)。
- **新seg_head(bagging+soft target+OOF三阈值)已回退**(commit 5f83c3b):8类实测3平5负
  (pcb 0.251→0.028崩塌),根因=OOF抛弃头阈值跨头迁移失败,绝对值/分位数两版迁移
  互斥失败。代码留在aoi/seg_head.py作opt-in研究件。
- crop_cascade(独立crop-head级联):ViSA pcb1实测-0.059,门控自动禁用。
- 按类选新旧seg_head的fit侧门控:CV原理上测不到fit/test漂移,3/3选错,已撤。
- 组件图"默认开":纹理类伪组件-0.218灾难;fit侧±0.1小信号边际增益估计=掷硬币。
- UniVAD v2(Hungarian组件匹配替代z-score)已判负(commit 44e07a7):同一fit受控A/B
  (排除训练随机性混杂)LOCO 5类均值Δ含漏检 v1=+0.016 vs v2=+0.004,理论动机(抓
  错序/缺件)未在真实数据兑现——单一全局刚性ECC warp限制了错序检测空间,"并集贴
  两块"策略判断不准时反而伤精度。默认回退v1,Hungarian代码留opt-in。
- UniVAD v3(局部搜索,试图解决v2判负点出的"全局刚性warp"问题)也已判负(commit
  5732111):同一fit受控A/B,LOCO 5类均值Δ含漏检 v1=+0.018 vs v3=-0.033,4/5类
  明显更差。冒烟测试就暴露信号(合并掩膜像素翻倍)——局部搜索的额外自由度(K组件×
  多候选偏移)带来更严重的多重比较问题,假阳性成本压过真实错位信号。教训:给逻辑
  异常检测加候选/自由度不是越灵活越好。默认关,代码留opt-in。至此UniVAD衍生的两次
  增量都判负,v1(严格门控)仍是唯一在产版本。
- DCP-SFR边界残差头已判负(commit 1f7495c):目标场景pcb/battery(本项目历史最弱
  两类)+AD2 sheet_metal三类验证,均值Δ纯定位=-0.052/Δ框=-0.069。pcb/battery门控
  正确拦截;唯一判"开"的sheet_metal fit留出gain=+0.074,test集实际只有-0.012/
  +0.043——fit侧正增益估计不可靠这条教训今天第三次应验。默认关,代码留opt-in。
- TTA(水平翻转取logit均值)已判负(commit ebd1ca6):5类均值Δ纯定位=-0.080/
  Δ框=-0.060,3/5类明显负(sheet_metal-0.173/walnuts-0.205/fruit_jelly-0.033),
  2/5类接近持平(pcb/battery)。这次不是"fit侧判不准"的老问题(TTA无需学习/门控,
  确定性),而是新教训:seg_head/WRN特征和标定统计量是在原始朝向图像上训练/标定的,
  不具备翻转不变性,喂模型从未见过朝向的镜像图产出系统性更差的预测,平均进去反而
  拖累原始预测。默认关,代码留opt-in。
- **WRN Conv-LoRA已判负,但措辞需精确**(commit见wrn_lora/,独立子工程未改aoi/):
  只在WRN layer2最后1~2个Bottleneck的conv2插低秩空间卷积旁路(BN全程冻结eval,
  权重可数学精确合并回普通conv、推理零增量延时),3类(sheet_metal/walnuts/
  fruit_jelly)×3种子×2配置(r2/r4)受控A/B,margin配对判定(median(Δ)>=0.005且
  > =2/3类中位数为正且min(Δ)>=-0.01)双双不通过:LoRA_r2 median(Δ)=0.000
  > (0/3类为正),LoRA_r4 median(Δ)=+0.001(2/3类为正,远低于0.005门槛)。封存前
  > 诊断排除"配置太保守/适配器没真动"的可能:保守配置下||ΔW||/||W_base||已达
  > 1.3~1.7e-3(远超1e-4死区);唯一压力测试(fruit_jelly r4/lr=1e-3/300步)证实
  > 权重可大幅移动(ΔW/W达4.5e-2)且fit/test同向大涨(+0.108/+0.114)。**准确结论:
  > fruit_jelly上确实可能存在类别特定的真实收益,压力测试没有排除这一点——不能说
  > "WRN表示已经足够"或"绝非超参数问题"。能确定的只是收益不广泛、不稳定(3类中
  > 2类稳定打平),继续在这个方向调参的竞赛期望值低。** 默认关,代码留opt-in研究件。
  > 最大价值:排除了一个"零时延但可能提分"的诱人方向,让资源能放心转向已有明显
  > 候选框信号的参考ROI。
- **重要修正(2026-07-29):上面的判负结论被证明是"配置太保守"的假象,不是机制本身
  不行**(wrn_lora/diagnose_aggressive.py + validate_production.py + gated_
  validate.py)。把压力测试用的激进配置(lr=1e-3/steps=300,原判负用的是lr=2e-4/
  150步)搬到全部AD2三类(sheet_metal/walnuts/fruit_jelly),3/3类全部转正
  (ΔIoU=+0.076/+0.030/+0.111);再搬到7个生产类目(hazelnut/cable/pill/carpet/
  leather/metal_nut/wood)重验(AD2结果不算精度证据,只能验证AD2上"确实是超参数
  问题"这个假说本身),ΔIoU=[+0.015,+0.011,+0.129,+0.183,-0.049,+0.049,+0.028],
  median=+0.028 mean=+0.052 min=-0.049,6/7类为正,但min没过-0.01门槛(leather
  fit涨test跌,典型过拟合)。**加OOF门控**(30张fit缺陷图内部切train_sub/val_sub,
  LoRA只在train_sub上训,val_sub上自检真的比base强才启用,否则退回base):门控后
  leather被正确拦截(Δ从-0.049变成0.000),7类结果=[+0.002,+0.061,+0.001,+0.177,
  0.000,+0.119,0.000],**min=0.000(零回退,没有任何一类变差),median=+0.002
  (差一点没过0.005严格线,主要是leather/wood两类门控生效/数据太少拉低了中位数,
  不是机制无效)**。已知局限:fit集本身很小时(cable/pill/metal_nut仅7~8张,wood
  仅3张)门控内部留出集(val_sub经常只有1~2张)统计意义存疑,判定噪声大。**保留
  该方向,列为P1候选**(净正+零回退,比其余9条候选证据都强),production集成前
  仍需:更多种子重复确认稳定性、小样本类目门控可靠性单独评估。
- **终局判负(2026-08-17):上面这条"配置太保守/加门控后净正"的修正结论本身也不成立
  ——WRN-LoRA在真正的生产管线里是净负的**(wrn_lora/eval_production.py)。此前所有
  LoRA数字(判负的、激进转正的、门控净正的)都出自wrn_lora/diagnose.py那套**独立
  测试台**:只算裸分割头IoU,不走det.locate()完整链路(没有图级is_defect门/SAM/
  框合并),也从没测过正常图假阳性率。补齐这两点重测,并且**发现并修掉一个致命
  实现bug**后,结论彻底反转:
  - **致命bug:LoRA训练时梯度被截断,全程空转**。训练用的`det.seg_head.extractor`
    就是`_wrn_feats`,而它带`@torch.no_grad()`(推理路径本来不需要梯度)——LoRA
    参数收不到任何梯度,up权重永远停在零初始化(=恒等映射)。Adam对`grad=None`的
    参数**静默跳过、不报错不告警**,所以整个流程"跑得很正常",跑出来的"提升"全是
    重训分割头带来的,和LoRA无关。修法:另写一个不带no_grad的等价前向
    (`grad_extractor`),并加硬断言(训练后LoRA权重变化量必须>1e-9,否则抛异常)。
  - **空转版(LoRA实际没生效)**:ΔIoU=[+0.006,+0.003,+0.001,-0.010,+0.015],
    median=+0.003 mean=+0.003 min=-0.010。
  - **真LoRA(梯度打通,权重确认变化6e-2~1.2e-1量级)**:ΔIoU=[+0.002,+0.002,
    +0.004,-0.015,**-0.047(phone_battery)**],**median=+0.002 mean=-0.011
    min=-0.047,不通过**。phone_battery从空转版的+0.015翻到-0.047,整体均值从
    正转负——**LoRA真正生效之后是主动帮倒忙,不是"没用而已"**。
  - 附带确认(这两点是干净的,可复用):①**推理零延时增量**(merged_conv()合并回
    普通卷积,4组背靠背实测差1~5ms,且merge前后输出数值完全一致);②**图级acc
    完全不受影响**(5/5类Δacc=0.000)——LoRA只改_bb_loc(定位骨干),而seg_head
    只在locate()判定is_defect=True之后才被调用,结构上不碰图级判定,没有GCAD
    那种假阳性风险。
  **结论:WRN-LoRA这条线彻底关闭,不要再重开。** 上面2026-07-29那条"修正"记录
  保留作为过程留档,但其结论已被本条推翻——教训是:**独立测试台的正向结果不能
  外推到生产管线,而且"训练跑通了"不等于"要训的东西真的在训"(必须加权重变化
  断言)**。
- **意外副产物(待验证,零成本候选)**:空转版那+0.003 median/+0.026框命中是真实的,
  它来自一个和LoRA无关的差异——生产`_seg_head_old_ae5fbbb.fit()`是把线性头和卷积头
  **各自独立训300步**(`_train_one(lin)`、`_train_one(cnv)`)再包成`_Ensemble`;而
  探针脚本是把`_Ensemble`当整体**联合训练**(梯度同时流过两个头,让它们协同)。
  pill的框命中提升(0.426→0.603,+0.177)在LoRA生效/不生效两次跑里**数字分毫不差**,
  说明确实是这个训练方式差异带来的。这个改动不碰骨干、零延时、零假阳性风险,是
  目前唯一未验证的免费候选,值得单独跑一次margin判定。
- **✅ 双头联合训练已验证并上生产(2026-08-17,今天唯一转正的候选)**
  (`seghead_tuning/probe_joint_ensemble.py`验证 + `_seg_head_old_ae5fbbb.py`的
  `joint_ensemble`开关,`CompetitionLargeDetector(joint_ensemble=True)`默认开)。
  改动只有一行逻辑:把`_Ensemble(lin, cnv)`当整体训300步,而不是`_train_one(lin)`、
  `_train_one(cnv)`各训各的——梯度同时流过两个头让它们协同分工。其余loss/优化器/
  步数/lr/batch/种子/归一化全部不变。
  - **探针6类**(隔离对照,含20张held-out正常图测假阳性):ΔIoU=[+0.006,+0.003,
    +0.001,-0.010(pcb),+0.015,+0.043(breakfast_box逻辑异常)],median=+0.004
    mean=+0.010 min=-0.010,框命中均值+0.018,**图级acc 6/6类完全不变**。
  - **官方成绩单5类**(`scripts/run_scorecard.py`,最终裁判口径):
    图级acc 0.909→**0.920(+0.010)**、含漏检IoU 0.506→**0.511(+0.005)**、
    框命中 0.554→**0.582(+0.028)**、纯定位IoU 0.527→0.525(-0.002,该指标
    不在赛题评分口径内)。**赛题评分的三项指标全部改善**,延时不变。
  - **注意**:逐类有正有负(hazelnut/cable的IoU跌了0.029/0.009,pcb反而涨0.047),
    是"均值小幅净正"不是全面碾压;而且**探针口径和官方口径的逐类结论差异很大**
    (pcb探针-0.010、官方+0.047),说明该改动对fit数据构成敏感,换批数据结果
    可能不同。幅度不大但方向一致且零成本,按"蚊子腿也是肉"保留。
  - `_select_feat_mode`里的临时头也同步传了这个开关,避免"用独立训的规格选特征
    模式、却用联合训部署"的规格不一致。
- **联合训练的步数扫描已判负,维持300步(2026-08-21)**(seghead_tuning/
  probe_joint_steps.py + _resume.py)。假说:300步/5e-3是当年针对"两个头各自独立
  训"调出来的,联合训练后可训参数翻倍、还要学协同分工,300步可能不够。只变步数
  (300/600/900),lr固定5e-3(避免像早前"激进配置"那样步数+lr一起变分不清谁的
  功劳),同一进程同一份fit特征训三个头,6类对照:
  | 类目 | 300 | 600 | 900 | 最优 |
  |---|---|---|---|---|
  | hazelnut | **0.643** | 0.591 | 0.630 | 300 |
  | cable | **0.812** | 0.785 | 0.800 | 300 |
  | pill | 0.469 | 0.480 | **0.486** | 900 |
  | pcb | 0.255 | 0.252 | 0.253 | 持平 |
  | phone_battery | 0.387 | **0.440** | 0.388 | 600 |
  | breakfast_box | 0.440 | **0.479** | **0.479** | 600/900 |
  - **600 vs 300**:ΔIoU median=+0.004 mean=+0.004 **min=-0.052**,
    **Δ框命中均值=-0.030** → 不通过
  - **900 vs 300**:ΔIoU median=-0.000 mean=+0.005 min=-0.013,
    Δ框命中均值=-0.018 → 不通过
  **判负理由**:①IoU中位数增益≈0,而**框命中全线下跌**——框命中正是联合训练
  当初最大的收益点(+0.028),加步数会把它吃掉;②min破线严重(600步hazelnut
  -0.052);③**最优步数逐类完全不同且形状非单调**(hazelnut是300好→600坑→900
  回升,phone_battery是600尖峰),这个模式说明**噪声大于信号**,不存在一个"更对"
  的固定步数。生产维持steps=300不变。
- **核心seg_head训练配方(300步/5e-3)本身也存在同样的"保守配置"模式,已验证并加
  OOF门控**(seghead_tuning/probe_aggressive_train.py + gated_train.py)。先测
  裸激进配置(900步/1e-2,全fit集直接训,不切分)在10类生产类目上的效果:ΔIoU=
  [+0.011,-0.007,+0.077,-0.008,+0.016,+0.007,**-0.089(wood)**,-0.026(pcb),
  +0.041,+0.052],median=+0.009 mean=+0.007 min=-0.089,6/10类为正但min严重
  破线——和LoRA同一个模式(多数类目受益,少数因过拟合大幅变差)。**加同款OOF
  门控**(fit集内部切train_sub/val_sub,两种配置都只在train_sub训,val_sub上
  自检选边)后:ΔIoU=[+0.023,-0.003,+0.065,0.000,0.000,0.000,-0.017(wood),
  +0.017(pcb),0.000,0.000],median=0.000 mean=+0.009 **min=-0.017(比未门控的
  -0.089收窄5倍)**,仍未过严格线。**唯一漏网的还是wood**:fit集仅3张,内部
  留出集只剩1张,门控判断在n=1上等同于抛硬币,这次判错(选了激进,实际变差)——
  不是门控原理有问题,是这一类数据量小到任何内部自检都不可能可靠。**结论:
  门控对fit集有基本量(观察到≥6张的类目门控都判对或安全退回)的类目可靠,建议
  加一条硬性下限(fit<5~6张时直接强制保守,跳过门控尝试),即可堵上wood这种
  数据量极端小的漏洞。** 和LoRA一样列为P1候选,production集成前需要补这条下限
  并重新验证。
- **RDDN-YOLO(参考图差分切片+YOLO候选框)冻结模型阶段测出候选框能力上界,尚未
  完成生产验证;LoRA微调阶段验证为负**(独立子工程rddn_yolo/,未改aoi/):ECC
  配准+Lab/梯度/局部SSIM差分通道构成6通道输入,在Real-IAD 12类phone/electronics
  类目上预训练nc=1 YOLOv8n(60epoch)。①冻结模型在phone_battery上的
  候选框召回@IoU0.3=0.700(vs朴素阈值基线0.280)——**⚠️这个阈值是在测试集本身上
  逐模型扫描出的最优值,只代表能力上界,不是生产可用的验证结果**(阈值必须只在
  fit/OOF数据上确定,测试集上扫阈值等于用了测试集信息)。②用赛题30张缺陷+100
  正常图对YOLO检测头做LoRA微调:修正"共用阈值"confound后仍为Δ=-0.062,LoRA
  微调方向不追。**要把冻结候选框从"能力上界"升级为"生产验证"必须补齐**:
  (a)阈值只在fit/OOF数据上确定,不摸测试集;(b)独立测试集上报告框命中率和
  严格IoU(不只是候选框召回);(c)包含低置信度回退全图WRN后的完整管线端到端
  结果,不是候选框proposer单独指标;(d)强制异常路径(非纯正常图捷径)测端到端
  p90延时;(e)同时报告该ROI分支的实际启用比例和每张图新增时延。这五条是Top-1
  参考ROI落地前的验收门槛,不是可选项。
- **Top-1参考ROI精修已按5条验收门槛真实验证,三条数据路线都没给出净正,暂不
  转正**(独立子工程rddn_yolo/roi_refine.py,未改aoi/):①Real-IAD pcb/
  phone_battery原生只有256×256(比WRN分割用的512还小),min_native门槛正确
  整体自禁用(0/40两类都是)——这个数据集本身没有分辨率headroom可供机制验证,
  不是bug。②MVTec LOCO(breakfast_box/juice_bottle/splicing_connectors,原生
  800~1700px,真有分辨率余量,只用structural_anomalies子集)3/3类OOF gain全负
  (-0.735/-0.289/-0.565),阈值扫描全收敛到网格下限0.05——YOLO是Real-IAD电子件
  上预训练的,LOCO是日用品,候选框置信度分布本身不可靠,fit()的OOF门控正确识别
  并全部禁用(enabled=False×3),test集结果原样未受影响(Δ=0.000,零回退纪律
  生效)。③pku_pcb(原生2000~3000px,真PCB,域匹配度最好,已在项目里用作大图
  延时探针)缺少clean/normal参考图,无法套用few-shot结构,未测——是真实数据缺口,
  不是判负。**结论:安全门控本身工作正常,但"同域+大分辨率+有normal参考"这三个
  条件本地数据凑不齐,至今未能在任何真实数据上验证出Top-1 ROI净正。** 默认不
  接入competition.py,代码留opt-in。下一步若要继续这个方向,需要找/补带normal
  模板的大分辨率同域(手机/电子件)数据,而非在现有三条路线上继续调参。
- **GCAD风格全局上下文分支已判负**(独立子工程global_context/,未改aoi/):动机是
  现有EAD/WRN/DINO门全是patch级局部判断,对"缺件/错位/组合"这类局部都正常、整体
  构图错了的逻辑异常结构性看不见(参照MVTec LOCO论文自己的baseline GCAD,加一条
  "整图过瓶颈,重建误差判构图对不对"的支路)。两个变体(PixelAE=忠实原论文的整图
  像素级卷积自编码器;EmbedAE=DINOv2 CLS token瓶颈重建,测语义嵌入是否有额外增益)
  ×9类目(5类LOCO logical_anomalies目标场景+2类LOCO structural_anomalies回归检查+
  cable/pcb 2类生产回归检查)充分对比:PixelAE median(ΔIoU)=+0.002 min=-0.021
  (juice_bottle逻辑异常);EmbedAE median=0.000 min=-0.014(screw_bag逻辑异常)。
  两者均远低于0.005门槛且跌破-0.01最差类别底线,即使只看5类目标场景中位数也只有
  +0.003,同样跌破底线。5/9(PixelAE)、4/9(EmbedAE)类目有正向移动(breakfast_box
  逻辑异常+juice_bottle结构异常两个变体都表现不错),但不足以支撑广泛稳定收益。
  **顺带回答了驱动这次实验的问题("语义容量是不是瓶颈"):EmbedAE中位数(0.000)
  并不明显优于PixelAE(纯像素,+0.002)——加更强语义表示本身不是瓶颈,"整图瓶颈
  重建判构图"这个架构思路收益本身有限,不是换更大模型能解决的。** 默认不接入
  competition.py,代码留opt-in研究件。副产品:意外发现`_calibrate_latency`硬线
  超时会砍DINO门这条路径在GPU高负载/高温时真实触发(9次里5次),已记入上方
  "cable问题根治记录"后的风险提示,待专门验证。
- **重要修正(2026-07-29):EmbedAE变体的判负结论被证实也是"配置太保守"**(global_
  context/eval_aggressive.py,steps 300→900/lr 1e-3→3e-3,和WRN-LoRA/seg_head
  同一个模式)。同9类目重测:PixelAE=[+0.021,+0.029,+0.010,-0.069,+0.002,-0.011,
  +0.004,0.000,0.000],median=+0.002 mean=-0.002 min=-0.069,**仍不通过**
  (screw_bag拖累);**EmbedAE=[+0.025,+0.058,+0.016,+0.004,+0.010,-0.003,+0.045,
  0.000,+0.007],median=+0.010 mean=+0.018 min=-0.003,7/9类为正,四项判据全部
  通过,不需要额外门控就直接合格**。原结论"加更强语义表示本身不是瓶颈"是在两个
  变体都训练不足时下的判断,不准确——练够了之后EmbedAE明显强于PixelAE(语义嵌入
  确实有价值)。**列为P1候选**,是目前3个"保守配置"候选里唯一不需要OOF门控就
  直接过严格margin判据的,production集成前仍需多种子重复确认稳定性。
- **重大教训(2026-08-13):GCAD-EmbedAE已真实接入生产`aoi/competition.py`又立刻
  回退,默认关**(代码留`aoi/gcad_embed.py`+`aoi/dino_gate.py`的`last_cls`缓存,
  opt-in)。接入方式是OR门(base=EAD+DINO该抓的原样保留,只有base判"正常"时才
  独立查EmbedAE阈值,复用DINO门同一次前向的CLS token,零增量前向)。**根因:
  当天所有验证脚本(eval_aggressive.py/eval_emb_prod5.py/combined_probe.py)
  都只在test_defs(缺陷图)上算IoU/hit,从没用正常图测过假阳性率**——"9类/5类
  不需门控直接过严格margin判据,min=0.000"这个结论只测了召回这一侧,完全是
  方法论盲区。真用`scripts/run_scorecard.py`(含正常图的完整图级acc)一测:
  **图级acc从0.902崩到0.703(-0.199)**,IoU/框命中只有+0.006~+0.022的小幅
  提升,完全不能抵消——OR门的独立阈值只要稍松,在"正常图占多数"的真实测试集上
  误报绝对数量就很大,不是只看缺陷图的判据能测出来的。立刻回退(`gcad_embed=
  False`),重跑scorecard确认5类全部干净恢复基线(均值acc=0.909,在正常波动
  范围内)。**这是今天最大的一条方法论教训:任何要接生产的候选,验证判据必须
  同时包含"正常图假阳性率"和"缺陷图召回率"两侧,只测一侧(哪怕min=0.000看着
  很稳)都可能是假象。** 后续若要重新推进,必须先补齐正常图OOF假阳性率验证,
  独立阈值不能直接套用EAD/DINO那套"分离正常/缺陷"的常规标定逻辑。
- **重要发现(2026-07-29):seg_head门控+GCAD-EmbedAE两个单独验证净正的候选,组合
  接入同一批生产5类目(hazelnut/cable/pill/pcb/phone_battery)后是净负**
  (combined_probe.py)。ΔIoU=[+0.031,-0.022,+0.066,-0.083,-0.053],median=
  -0.022 mean=-0.012 min=-0.083,不通过。对比单独测试:cable单独seg_head门控
  -0.003/单独GCAD从未触发,组合-0.022;**pcb单独seg_head门控+0.017/单独GCAD
  +0.007,组合-0.083(反转成大幅负)**;phone_battery单独seg_head门控0.000/单独
  GCAD+0.007,组合-0.053。**不是简单的效果相加,是真实的负向交互**,大概率是两个
  机制都各自在同一批很小的fit集上独立做阈值/门控标定,叠加后标定噪声被放大而不是
  信号被放大。**结论:两个候选目前都不能直接合并接入生产**,即使各自都已验证净正
  也不行——组合效果必须专门验证,不能假设可加性。根因(是否该共用同一次内部留出
  集标定而非各自独立切分)留作后续课题,不在本次范围内深挖。LoRA门控未纳入本次
  组合测试(和_select_feat_mode有耦合风险,见上,待单独确认安全后再测组合)。
- **上面的"负向交互"被查明是测试脚本自己的bug,不是两个机制真的互相拖累
  (2026-07-29,追加排查)**:`pick_gated_head`(seghead_tuning/gated_train.py)
  门控决策用train_sub/val_sub切分本身没错,但**门控选中的配置(不管保守还是
  激进)最终只在train_sub(约70%fit集)上训**,而组合测试的baseline对照用的是
  `det.fit_fewshot()`正常流程、**用了全量fit集**训出来的seg_head——不是公平
  对比,门控哪怕选中"保守"也因为少用30%数据而天然更弱。证据:phone_battery
  门控选了"保守",理论上该和baseline打平,组合测试却是-0.053。**修复**:决策
  阶段仍用train_sub/val_sub(便宜),但决策完成后用全量fit_i/fit_m重新训一遍
  胜出的配置,保证和baseline数据量对等。修复后重跑:
  - **单独seg_head门控(10类)**:ΔIoU=[+0.008,-0.010,+0.076,0.000,0.000,0.000,
    -0.001,-0.016,0.000,0.000],median=0.000 mean=+0.006 min=-0.016(wood从
    -0.089/-0.017大幅改善到-0.001,证明修复对症;pcb仍-0.016,说明pcb是
    "激进配置全量重训后依然过拟合"这个类目本身的问题,不是数据量bug)。
  - **组合(seg_head门控+GCAD-EmbedAE,生产5类)**:ΔIoU=[+0.044,-0.012,+0.077,
    -0.032,+0.021],**median=+0.021 mean=+0.020 min=-0.032,方向从净负彻底
    转正(3/5类正,phone_battery从-0.053翻正到+0.021,pcb从-0.083收窄到
    -0.032)**,仍未过严格min≥-0.01门槛,但已经证明"负向交互"这个结论是假的,
    真实情况是"组合本身基本安全,唯一剩下的问题是pcb这一类对激进训练本身敏感"。
    **后续:pcb需要单独处理(排除出激进候选名单,或专门诊断这类的过拟合成因),
    其余4类已经具备净正证据。**
- **AHL(Anomaly Heterogeneity Learning,CVPR2024,赛题唯一点名参考文献)适配
  seg_head训练过程已判负**(独立子工程ahl_seghead/,未改aoi/):动机是这篇文献
  方法论上最贴合赛题场景(少量标注异常+需泛化到训练时没见过的异常类型="open-set
  supervised anomaly detection"),且推理零增量延时(论文原文只用训好的统一模型
  推理)。适配思路:fit期正常图聚类(k-means,C=3)→每次随机取一簇正常+随机70%
  标注缺陷组成"异态代理集"(support/query各切一半,重复T=7次)→每个代理集
  support上各训一个基础头(和生产SupervisedSegHead同架构)→在全部代理集的query
  集(对训出该代理集基础头是"没见过的")上协同训练**一个统一头**,只有这个统一头
  用于推理,替换进`det.seg_head`后复用`det.locate()`完整下游评测。3类摸底
  (breakfast_box逻辑异常/pushpins逻辑异常/cable生产回归检查)**3/3全负且方向
  一致**(ΔIoU=-0.037/-0.039/-0.015,median=-0.037 mean=-0.030)——不是GCAD那种
  类目间正负交替的噪声信号,是干净一致的负结果,没有扩到9类目普查的必要。
  **最可能根因**:我们的fit缺陷集本来就极小(15~30张),这套机制的核心动作是把
  这个已经很小的集合再切成T=7个更小的support/query子集训练——论文场景(DevNet/
  DRA分类式打分网络)下这样切片能提升泛化,但对我们这种逐像素分割任务、标注样本
  本来就少到individual instance都数得过来的场景,再切片等于让每个子集看到的
  独特缺陷实例更少,"人为制造开放集泛化信号"的收益没能盖过"可用监督信号被进一步
  稀释"的代价。**不代表AHL思路在更大样本量场景下没用,是"30张缺陷"这个量级本身
  可能撑不起论文假设的"充分切片仍有信号"这个前提。** 默认不接入competition.py,
  代码留opt-in研究件。
- **统一骨干提案(用户提出:一骨干四头+多教师蒸馏,替代EAD+DINO+WRN三模型三特征
  空间)的两条低成本探针均判负**(独立子工程unified_student/,未改aoi/):
  ①图级判决蒸馏探针(`probe_distill.py`)——测"复用现有冻结WRN特征能不能替代
  EAD+DINO联合判决",mean/top-k两版聚合方式都试了,一致率median分别0.724/0.674,
  远低于0.90门槛,且按类目分化(局部缺陷类top-k更好,逻辑异常类top-k反而更差)。
  最damning:pcb上mean pooling相关系数只有0.068,cable上学生漏判DINO独立救回的
  那部分缺陷。②SAM实例计数探针(`probe_sam_counting.py`+`_defect.py`)——测
  "MobileSAM everything模式计数能不能捕捉逻辑异常",地基(计数稳定性)4/5类没问题,
  但**everything模式延时5.9~7.2秒/张,是190ms预算的30~36倍**,决定性判负,区分度
  本身也不稳定(juice_bottle 9/10张明显,其余3类0~2/10)。**结论:这两条低成本捷径
  (复用现成特征、复用现成SAM)都不可行,不代表"专门蒸馏新骨干+真实标注监督"这个
  更大的方案本身失败——那需要真正投入建才能验证,今天没做,只是排除了两条更便宜
  的替代路线。** 默认不接入competition.py,代码留opt-in研究件。
- **FocalLoss+DiceLoss替换seg_head的BCE+pos_weight已判负**(独立子工程
  focal_dice_seghead/,未改aoi/;借鉴自MultiADS论文的训练配方,FocalLoss/DiceLoss
  本身是通用技术非其原创,重新推公式实现,不涉及MultiADS的AGPL版权问题):5类真实
  数据(pcb/phone_battery/cable/breakfast_box逻辑/pushpins逻辑)ΔIoU=
  [-0.039,-0.016,+0.009,+0.032,-0.024],median=-0.016 mean=-0.0076 min=-0.039,
  2/5类为正,不过关,中位数均值都是负的。**反直觉**:理论上FocalLoss该救的微小
  缺陷类目(pcb/phone_battery)跌得最狠,反而在breakfast_box(逻辑异常,不是微小
  缺陷)上涨最多——"损失函数换成FocalLoss能救微小缺陷"这个理论预期没有兑现,
  具体原因未深究(可能WRN特征表达能力才是瓶颈,损失函数怎么调都够不着;也可能
  alpha/gamma超参没针对我们数据调过,直接用了MultiADS默认值)。默认不接入
  competition.py,代码留opt-in研究件。
- **FocalDice的alpha反向假说已证伪(2026-07-29)**(focal_dice_seghead/
  eval_focal_dice_alpha75.py):怀疑alpha=0.25(降权正样本)方向选反了,微小缺陷
  该加权正样本才对,换alpha=0.75重测同5类:pcb=-0.053、phone_battery=-0.041、
  cable=+0.008、breakfast_box=+0.045、pushpins=-0.043,median=-0.041 mean=
  -0.017 min=-0.053,**比原alpha=0.25那次(median=-0.016)更差,两个微小缺陷类目
  跌得更狠**。结论:不是alpha方向问题,原判负结论维持,且进一步确认FocalDice
  这套损失函数在我们的极端类别不平衡场景下没有可调参数能救,不用再往这个方向试。
- **顺带一提**:调研MultiADS(ICCV2025,boschresearch/MultiADS,AGPL-3.0协议)
  确认它的核心视觉机制(CLIP+linear adapter做逐像素密集预测)和已判负的
  AnomalyCLIP是同一路数,"多类型缺陷同时检测"这个卖点在MVTec/MPDD/Real-IAD等
  数据集上实际也是"每张图仍是单一缺陷类型",和我们现有"色彩/尺寸/结构3分支
  z-score竞争决定类型"这套设计本质上是同一个思路的另一种实现(CLIP文本相似度
  vs 手工统计z-score),不是全新架构。唯一真正借鉴到并验证过的是上面的
  FocalLoss+DiceLoss训练配方(已判负)。
- **便宜版连通域计数(ECC模板残差,非SAM)融合进图级检测已判负**
  (unified_student/probe_residual_gate.py,未改aoi/):SAM everything模式计数
  延时判死后,改用crop_cascade.py已有的ECC对齐模板残差(Lab色差+梯度差→连通域,
  ~20-40ms量级,不是SAM的秒级开销)当第三个图级信号,融合进max(z_EAD,z_DINO,
  z_residual)。5类验证(breakfast_box/pushpins/screw_bag逻辑+cable/pcb回归)
  ΔIoU=[0.000,-0.020,+0.032,0.000,0.000],median=0.000 mean=+0.003 min=-0.020,
  不通过。3类是0.000(残差信号从没比EAD+DINO更高,根本没触发过判决变化),
  screw_bag唯一真正生效且为正,pushpins生效但为负。**结论:这个信号大部分时候
  太弱,压不过EAD/DINO,偶尔触发时正负各半,没有稳定收益。** 默认不接入
  competition.py,代码留opt-in研究件。
- **肉眼失败案例诊断(unified_student/visualize_failures.py,2026-07-27)**:挑
  breakfast_box的logical_anomalies测试图肉眼检查(不是新机制验证)。6张图4中2
  漏(009/010漏检)。关键发现:漏检的009(GT前景246346px)和命中的019(242772px)
  标注面积几乎相等,不是"缺陷小才漏检";raw EAD分数连续谱上漏检两张
  (0.633/0.635)紧贴命中中最弱的一张(0.654),不是分数骤降或找错位置;命中样例
  的预测掩膜大体落在真实缺陷区域(坚果混入granola格子),只是水果纹理反光带来
  散点FP。**结论:这是信号强度不够、卡在阈值边缘的margin问题,不是掩膜定位错
  地方或存在结构性盲区的bug**——和当天8条机制路线全部判负互相印证:现有EAD+
  DINO局部patch比对范式,对"轻度构成性替换/欠量"类逻辑异常原始信号本来就弱。
  仅n=2漏检样本的定性观察,不构成可直接改动生产的证据,只作诊断记录。
- **INP-Former(CVPR2025,MIT协议,github.com/luow23/INP-Former)融合进图级检测
  已判负,且是本session最差的一次**(inp_former_probe/probe_inp.py,未改aoi/;
  官方Few-Shot(k=4) MVTec-AD checkpoint直接可测,hazelnut/cable/pill/carpet/
  leather/metal_nut/wood都在其训练用的标准15类里,无需自己训练):机制是每张图
  从自身patch分布提取"内在正常原型"做重建,误差当异常分数,融合进max(z_EAD,
  z_DINO,z_INP)。7类验证ΔIoU=[0.000,-0.111,0.000,0.000,-0.130,0.000,-0.143],
  median=0.000 mean=-0.055 min=-0.143,四项判据全部不过关。**不是中性无效果,
  是真实倒退**:cable/leather/wood三类接入后把原本正确的EAD+DINO判断压过去,
  分数实打实变差。初步归因:INP-Former给出的分数尺度/置信区间没有针对我们的
  融合场景校准过,直接max()等于让一个未经消化的外来信号越权推翻两个已大量调优
  的信号源的判断——本session另几条判负路线(ECC残差、蒸馏探针)也不同程度暴露
  同类问题,"外来信号直接max()进决策"这个模式本身可能需要重新审视,不只是逐个
  信号源试错。默认不接入competition.py,代码留opt-in研究件。
- **今日阶段性小结(2026-07-27)**:当天累计验证WRN-LoRA/Top-1参考ROI/GCAD全局
  分支/AHL异态代理集/图级蒸馏探针/SAM实例计数/FocalDice损失/ECC残差图级信号
  共8条独立路线,**全部未能broadly过关**(个别类目偶有正向离群点,但没有一条
  中位数稳定为正)。这个密度的一致性负结果本身是个信号:现有EAD+DINO这套局部
  patch级比对逻辑,在逻辑异常(缺件/错位/组合)这类问题上的天花板,可能就是当前
  测出来的量级(0.19~0.35 IoU),不是"还没找到对的改法"能简单解决的,继续在
  现有骨干上加信号/加分支/换损失函数,边际收益递减、风险递增。如果还要往这个
  方向投入,大概率需要真正投入建一个新骨干(离线多教师蒸馏+真实标注监督,而非
  在现有骨干输出的特征之上做文章),这是比今天任何一条都大得多的工程量,需要
  谨慎评估时间预算后再决定是否启动。
- 更早:DINO/SubspaceAD定位、AnomalyCLIP融合、RAMS-R上生产、CutPaste合成、
  roi_zoom原版——全负,勿重开。
- GPU"脏卡"陷阱:连轴跑后SM降频,延时读数系统性偏高;测延时前查nvidia-smi温度/频率。

## 待办(优先级序,2026-07-25更新——GCAD全局分支已封存判负,新增DINO门风险排查)

1. **排查`_calibrate_latency`硬线超时砍DINO门的真实触发概率**(今天意外发现,见
   cable根治记录后的风险提示):9次fit里5次在GPU高负载/高温下真实触发,评委机器
   若也处于高负载/高温状态,cable这类靠DINO门救回来的类目有回归旧问题的风险。
   需要专门验证不同GPU状态下探针读数的稳定性,评估是否要给DINO门更高的免砍优先级。
2. **数据缺口:找/补带normal模板的大分辨率同域(手机/电子件)数据**——Top-1参考ROI
   在Real-IAD pcb/phone_battery(256×256,无headroom)/MVTec LOCO(800~1700px但
   域不匹配,3/3类OOF全负)/pku_pcb(2000~3000px真PCB但无normal图)三条路线上都
   没能验证出净正(见"重大负结果"Top-1参考ROI条目),根因是本地没有同时满足"同域+
   大分辨率+有normal参考"的数据,不是机制本身被证伪。找到/补齐这类数据前,不建议
   继续在现有三条路线上调参。
3. **pcb/battery微小缺陷仍是主拉分点**(0.26/0.37量级)——DCP-SFR边界残差、UniVAD v2
   Hungarian、WRN-LoRA、GCAD全局分支、AHL异态代理集五条路线都已验证判负(见"重大
   负结果"),这几类目前没有已验证的改进候选在手,需要重新想机制而非在现有分割头
   上加小修正
4. TF-IDG生成增广:官方代码github.com/rubymiaomiao/TF-IDG,需>8GB VRAM机器跑生成
   (AnyDoor ckpt+DINOv2 ViT-G),本机侧门控评审脚本scripts/run_tfidg_gate.py已就绪
   (3切分OOF均升+最差类回退≤0.01+真实占比≥50%三条全过才准入)
5. Boxes2Pixels(低优先级/待定):仅当官方30张缺陷标注只有框、没有精确掩膜时启用
   (SAM出伪掩膜训紧凑学生+单向自纠正);标注格式未知前不必先实现
6. 2060真机延时验证:scripts/run_2060_check.py(合成图,免数据集);4060L(256GB/s)
   悲观代理+2070(448GB/s)乐观代理可夹逼2060(336GB/s)
7. GitHub push(repo Surigier/aoi_inspect私有,token用时向用户要)

**不建议做**(已讨论/已负结果,勿重开):RadioCore/FastRef/SubspaceAD/O2MAG——VFM榜单
强不等于PCB严格IoU强已实证;FastRef测试时优化增加热路径且面向少样本,契合度低;
SubspaceAD已负(含漏检IoU 0.484→0.462)。

## 关键脚本

- scripts/run_scorecard.py:5类全量生产成绩单(真口径,最终裁判)
- scripts/run_seg_head_ab_scorecard.py:新旧seg_head归因A/B
- scripts/run_2060_check.py:租卡/新机延时+显存一键验证
- scripts/run_pareto_scan.py:students×DINO×SAM×max_pixels Pareto扫描
- scripts/run_comp_graph_ab.py / run_comp_graph_harm.py:组件图A/B与伤害检查
- scripts/run_tfidg_gate.py:生成增广三条门控评审
- 实验日志写\_logs/(gitignore),用nohup跑长任务(会话重启不陪葬)

## VLM类型归属:双图对比是结构性突破(2026-08-22验证)

**结论:单图77% → 双图对比87%**(5类目×12张=60张,GT掩膜,不需要fit检测器)。

演进:启发式50% → 修自匹配标定bug 50% → 位置匹配手工特征58% → VLM单图v1 68%
→ v2 75% → v3 57%(推翻重写,判负) → v4 77%(v2+定点补丁) → **v5双图对比87%**。

**为什么双图是结构性的、不是提示词调参**:赛题5类里"缺件少件/尺寸偏差/逻辑错误"
这三类**本质上是比较出来的**——没有参照物,VLM根本无法判断"这里本该有个东西"。
实测cable缺线被判成色彩变化,就是因为它只看到"这块颜色不一样"。给一张裁在同一
坐标的正常参考图后:**cable 5/12 → 9/12**,pill 10→11,metal_nut 7→8,hazelnut保持12/12。

**对真实赛场更有利**:本测试里hazelnut/carpet的物体姿态是随机的,参考图裁同一坐标
其实对不齐,仍然没掉分;而赛题是固定治具上的手机部件,姿态一致,对齐质量只会更好。

**metal_nut剩余4张错判是基准标签噪声,不是模型问题**(已核实,不要再调提示词):
让VLM只描述物理现象,恰好4张回答"裂纹/剥落/材质断裂",和判错的4张一一对应。
MVTec把该类目命名为color,但图里确实是材料破损。**继续在这上面调提示词等于教模型
把裂纹叫成色彩变化,是真的过拟合。**

工程要点:`_crop_at()`必须用`_crop_with_box()`返回的同一组坐标裁参考图;
`normal_ref=None`时自动退回单图模式;无key/超时全部吞掉返回None→降级启发式。

## VLM类型头已上生产:端到端88%(2026-08-22)

`aoi/type_head.py` + competition.py三处接线。**fit期(不计时)用VLM给30张缺陷图打
5类标签 → 蒸馏成质心表;推理期零API、零外网。**

**端到端真数(预测掩膜,不是GT掩膜,scripts/eval_type_head.py)**:

| 类目 | VLM类型头 | 启发式(原) |
|---|---|---|
| hazelnut | 11/11 | 9/11 |
| pill | 12/12 | 1/12 |
| carpet | 12/12 | 4/12 |
| metal_nut | 4/11 | 0/11 |
| cable | 12/12 | 9/12 |
| **合计** | **51/58 = 88%** | **23/58 = 40%** |

端到端88% > GT掩膜离线上界87%,不矛盾:单产品的fit缺陷标签常常只有一类,此时
质心分类**退化成恒定输出该类**,反而对掩膜质量完全免疫。

**踩过的坑,别再犯:**
1. **"标签只覆盖1类就弃用整个头"是错的。** 单产品迁移场景下缺陷类型天然集中
   (hazelnut 17/17张全是常见外观缺陷),此时恒定输出该类正是最优估计。第一版
   写了`len(set(y))<2 → return False`,hazelnut直接从11/11掉成走启发式的8/11。
2. **参考图深层特征要1.5GB**(768×128²×30张),2060装不下 → 缓存降到32²粗格(94MB)。
   掩膜是块状区域,32²的空间分辨率足够做"在这块位置上池化"。
3. **推理期对30张正常图各跑一次WRN = 270ms,直接爆预算** → 全部挪进fit期预缓存。
4. `_wrn_feats`加了**单次locate内缓存**(seg_head和类型头共用,省9ms),在locate()
   入口显式清空——**不能靠data_ptr判等**(张量释放后地址会被复用,可能假命中)。
   type_head的fit循环里必须手动清缓存,否则30张参考图全拿到第一张的特征。

metal_nut 4/11是已核实的**基准标签噪声**(MVTec该类目名为color但图里确实是材料破损),
不是模型问题,不要再调提示词。

**降级链**:无key/无网/超时/标注失败/推理期异常 → 任一环断开都退回`_ztype`启发式,
检测与定位完全不受影响。评委机器无外网可直接运行。

## 2500²拼接大图压测 + 手机数据集现状(2026-08-24)

### 延时:2500²真尺寸下 中位97ms / p90 99ms / 最大105ms @4060L,预算200ms
`scripts/eval_phone_stitch.py`,4张手机屏各放大到1250²摆2×2拼成2500²,compile_infer=True,
预热3张后计时。**这是目前唯一一个在赛题真实输入尺寸上测出来的延时数**,此前所有延时
都是小图(640²/320²)测的。2070那轮跑完可与4060L夹逼估2060。

### 象限命中不均匀(待复测,别当结论)
左上14% / 右上50% / 左下5% / 右下8%。这是拼接图独有的失效模式(单图测不出来),
但每象限只有~20个框、且底层是7张正常图饿出来的模型,**样本量和模型状态都不足以下结论**。
有了正常图的干净数据必须复测这条。

### 手机数据集:两份都没有正常图 —— 这是硬阻塞
- `data/phone`:Roboflow YOLO,2012/287/576张,640²,oil/scratch/stain。**正常图仅7张**
  (还是2张原图的增广重复)。
- `data/phone_best`:960/120/120张,640×360,同3类,**正常图0张**。无增广重复,源图更干净。
- 两份都是**北大MSD数据集**(jianzhang96/MSD)的衍生版。原版1200张1920×1080,
  **官方另有 good.zip(27.7MB,20张无缺陷图)**,两份衍生版都把它剥掉了。
  下载:https://github.com/jianzhang96/MSD (OneDrive链接在README表格里)

**饥饿后果实测**:7张正常图fit → 640²单图 检出率88%但**框命中0.000**;2500²拼接
检出率75%/框命中0.158。GT框一点都不小(256²掩膜上中位35×50px、占图2.7%,无一低于
生产min_area门槛),所以0.000不是"缺陷太小",是模型没见过足够多正常、把大片区域
误标成异常。**这两个数只说明"正常样本饿到7张会崩",不代表定位能力,不要拿去汇报。**

**这两份数据真正的价值是类型归属验证(不需要正常图)**,已完成:
划痕→常见外观缺陷 20/20一致;污渍→色彩变化 18/19;油污分裂(色彩变化60%/外观40%)。

### 结论:正常图的答案本来就在硬盘上 —— Real-IAD
`data/_dl/Real-IAD/` 有**12个类目、每类约2400张OK图**、且是**真实像素掩膜**不是框:
audiojack / button_battery / end_cap / **pcb** / **phone_battery** / regulator /
**sim_card_set** / switch / terminalblock / transistor1 / usb / usb_adaptor
**现有成绩单只用了其中2个(pcb、phone_battery)**。`prep_realiad(cat)`是全参数化的,
另外10个直接就能跑。要做手机部件域的真实成绩单,这才是路子,不是去补MSD那20张。

## 迁移到2070怎么做(2026-08-24)

**分两路走,别整个拷**:这台机器上 aoi_inspect 连数据 119G,真正跑得起来只要 ~13.5G。

| 走哪 | 内容 | 大小 |
|---|---|---|
| **GitHub** | 代码全部(仓库本身就是交付物) | ~3MB |
| **U盘** | `backbones/`(DINOv2 **331M,超GitHub单文件100M上限**)、`models/`、`data/_dl/Real-IAD`、`data/_dl/realiad_jsons`、`.env` | 2.4G |
| U盘(完整档再加) | `data/mvtec`、`data/_dl/mvtec_loco`、`data/phone*` | +11G |

**明确不带(省105G)**:`rddn_yolo/`(22G,另一个YOLO项目)、`data/mvtec_ad_2`+tar(62G)、
`visa`/`dagm`/`mpdd`(6.5G,没进任何成绩单)、各种已解压的`.zip`/`.tar`(15G纯重复)。

一键脚本:`bash scripts/migrate_to_usb.sh <U盘路径> [--min]`

**到2070后**:
```
git clone https://github.com/Surigier/aoi_inspect.git && cd aoi_inspect
rsync -a /媒体/U盘/aoi_payload/ ./
pip install -r requirements.txt && python -m pytest -q     # 应 62 passed
PYTHONPATH=. python scripts/eval_phone_stitch.py           # 2070的2500²延时
```

**安全**:`.env`(DASHSCOPE key)在`.gitignore`里,已用`git check-ignore`确认;
全仓库+git历史扫过`sk-[0-9a-f]{32}`,零命中。key只走U盘,永远不进git——本仓库要交评委。

## 2070机器迁移完成 + 三档硬件延时(2026-08-24)

**机器**:192.168.10.140 / slam,**RTX 2070 SUPER 8GB**(不是普通2070),驱动580.82.09,
Python 3.8.10,torch 2.4.1+cu121。公钥已装(免密ssh)。仓库在 `~/yolo/aoi_inspect`,
数据2.6G(权重+Real-IAD+手机数据+.env)。`pytest -q` **62 passed**,与4060L一致。

### 迁移踩的三个坑
1. **那台机器连不上HuggingFace**,timm的WRN50/DINOv2骨干下不下来。把本机
   `~/.cache/huggingface/hub/models--timm--*` rsync过去(约350M)+设 `HF_HUB_OFFLINE=1`。
   **这是真实交付风险**:评委机器很可能也无外网,现在这版会直接起不来
   → 需补"离线部署"说明,或把timm缓存打进交付包。models/里只带了EAD教师和MobileSAM。
2. **rsync没传完就启动了任务** → HF缓存里是断链symlink指向rsync临时文件(`.xxx.G6JiYr`),
   报LocalEntryNotFoundError。校验方法:比对两边 `find -type f -exec du -b` 求和。
3. **`pgrep -f`/`pkill -f` 会匹配到自己那条命令**。用`pkill -f`时把执行命令的shell自己
   杀了(exit 144);用`pgrep -f`做"防重复启动"检查时误判成"已有实例在跑"。
   → 远程操作一律用 `ps -eo pid,cmd | grep [e]xact` 看实际进程,不要凭计数。

### 三档硬件的 2500² 延时(scripts/eval_phone_stitch.py,同一脚本同一数据)

| | 4060L (Ada) | 2070S 原生 (Turing) | 2070S@1260MHz(2060代理) |
|---|---|---|---|
| 中位 | 97ms | 129ms | 见下轮 |
| **p90** | **99ms** | **131ms** | 见下轮 |
| 最大 | 105ms | 139ms | |

**必须先纠正一个旧假设**:此前记的"4060L/2070夹逼2060"**不成立**——2070S和4060L
都比2060快,夹不住。改用**降频代理**:2070S和2060**同为Turing架构**(4060L是Ada,
跨两代不可比),把2560核锁到1260MHz使FP32吞吐≈2060的6.45 TFLOPS。
`scripts/lat_2060_proxy.sh`(带trap自动恢复)。频率1260正好是该卡支持的档位之一。

**代理仍然偏乐观,真2060只会更慢**,汇报必须标注四条:
①SM分布不同(2070S更多SM跑低频,2060更少SM跑高频,后者调度效率略低)
②显存带宽降不下来(448 vs 336 GB/s,`-lmc`消费卡多不支持)
③2060只有6GB显存,2070S有8GB,显存压力代理不了
④**热**:原生跑到84°C时频率1875→1860轻微降频;锁频轮65°C不降频。真2060(尤其
  笔记本/小机箱)更容易热降频,而赛题要连续跑1000+张。

**另一个观察**:我们这套管线把2070S吃到99%占用、84°C——不是轻负载。
持续满负载的热漂移必须单独验,不能只看短跑。

## 重大发现:2500²下延时**不是GPU算力瓶颈**(2026-08-24)

四点实测,同一脚本同一数据(scripts/eval_phone_stitch.py):

| 2500² 配置 | 中位 | p90 | 最大 | 跨度 |
|---|---|---|---|---|
| 4060L 原生 | 97 | 99 | 105 | 8 |
| 2070S 原生(下限可掉到300MHz) | **129** | **131** | 139 | 10 |
| 2070S @1875锁频 | 113 | 118 | 149 | 36 |
| 2070S @1260锁频 | 113 | 114 | 115 | 2 |

**核心频率砍33%(1875→1260),中位数一模一样(113ms),p90只差4ms。**
→ 这条管线在2500²下**根本不是GPU算力瓶颈**。

**赛题结论:2060守住200ms没有悬念**。不是因为余量大,而是**换更慢的GPU本来就影响不大**。
原先担心的"2070S比2060快1.5×所以要打1.5×折扣"——**这个折扣不存在**。
估计区间115~135ms(取决于评委机器是锁频还是自由boost)。仍需打折的只剩显存带宽
(2060 336 vs 2070S 448 GB/s)和6GB显存。

**原生反而最慢(129/131)**:频率下限自由掉到300MHz时,GPU在Python侧空隙里降频,
再boost回来的爬升时间被白吃。证据在分布跨度:原生10ms(boost抖动)、1260锁频仅2ms(钉死)。

### 【待证实的推断,别当结论】瓶颈可能在CPU/Python侧
2500×2500的numpy/cv2操作(插值、连通域、掩膜重采样、SAM后处理)很贵。若坐实,
**优化方向要整个掉头**——一直以来都在抠GPU侧(换骨干/减前向/加缓存),真正的杠杆
可能在CPU侧。测法:locate()里插`torch.cuda.synchronize()`分段计时,拆开GPU段与CPU段。

### 踩坑记录:我把这个代理实验做错过一次
原以为`nvidia-smi -lgc 1260,1260`只压峰值,**忘了它同时把频率下限从300MHz抬到1260**。
导致"降频反而更快"(114ms vs 原生131ms)的荒谬结果,一度当成2060的估计值。
**一次改了两个变量**。补1875锁频对照组(同样锁住下限)后才隔离出频率的真实影响≈0。
教训:锁频类实验必须设同下限的对照组。

### 本机(4060L)Python环境有损伤,打包前建议重装
连续撞到`TypeError: 'str' object is not callable`、`attribute name must be string`、
`bad BUILD_CONST_KEY_MAP keys argument`、`unsupported operand +=: int and NoneType`
——全在**导入阶段**、每次报错不同、重试即好。2070那台一次没出过。
**若最终打包在本机做,先重装环境**,否则可能打出坏包还查不出原因。

## 【重要教训】WRN特征缓存毁掉定位,以及我如何验证失败(2026-08-24)

### bug本身
给`_wrn_feats`加了"单次locate内缓存"(想让seg_head和类型头共用一次前向,省9ms),
缓存只在`locate()`入口清空。**漏了fit期**:`seg_head.fit()`会逐张调同一个函数,
而fit期没有locate()来清缓存 → 30张缺陷图全部拿到第一张的特征,配各自不同的掩膜
训分割头 → 头是垃圾。

**表现极具迷惑性**:图级acc几乎不变(0.925→0.912,判决不依赖掩膜),但
含漏检IoU 0.399→**0.033**、框命中 0.550→**0.013**。

修复=**彻底移除缓存**(不是打补丁)。省的9ms在200ms预算里毫无意义,而风险是静默
摧毁定位。理由与现象已写进`_wrn_feats`的docstring防止再犯。
修复后确认恢复:evaluate路径 acc=0.912/含漏检IoU=0.406/框命中=0.463。

### 我在这件事上连犯三个错(比bug本身更值得记)
1. **改生产代码只验证新功能(类型归属88%),没重跑权威成绩单** ——
   和GCAD-EmbedAE那次一模一样,而那条教训我当天下午刚亲手写进结果报告。
2. **"验证修复"时没确认验证真的执行了**。`run_scorecard_realiad12.py`有个
   "已有结果就跳过"的续跑缓存,读到了修复前的陈旧json直接跳过,把旧结果原样重放。
   我看到"数字没变"就断言"缓存不是根因"。**续跑缓存对续跑是好事,对验证是陷阱**
   ——已加`--fresh`强制重算。
3. 据此推断出"系统存在随机双峰、赛场只跑一次全看运气"这么严重的判断并上报,
   浪费约两小时机时。

**规约补充**:
- 任何生产代码改动 → **必须**重跑`run_scorecard.py`(含正常图)才能声称无回归。
- 任何"验证修复"的运行 → **必须**先确认它真的重新计算了(看日志里有没有
  "跳过"、比对耗时/时间戳),不能只看数字。

### 受此bug污染、需要重测的结论
- 类型归属端到端88%:当时吃的是坏掩膜(结果偏保守,重测应≥88%,但必须重测)
- 手机数据框命中0.000(640²单图)/0.158(2500²拼接):**作废**。我当时归因为
  "正常图只有7张",归因错误——主因是本bug。
- 2500²延时四点表:延时不依赖掩膜正确性,基本有效;但坏掩膜会改变SAM/框的工作量,
  理想情况下应复测一点确认。

## Real-IAD 12类目成绩单(2026-08-24,2070S,--fresh强制重算)

`scripts/run_scorecard_realiad12.py`,口径直接import run_scorecard的evaluate/prep_realiad。
100正常+30缺陷fit,40缺陷+40正常测。**这是目前覆盖面最广、最贴近赛题隐藏域(手机/电子
部件)的一份真口径证据。**

| 类目 | 图级acc | 含漏检IoU | 框命中@0.5 | locate |
|---|---|---|---|---|
| sim_card_set(SIM卡座) | 0.925 | 0.532 | **0.700** | 152ms |
| phone_battery(手机电池) | 0.912 | 0.402 | 0.512 | 171ms |
| terminalblock | 0.887 | **0.547** | 0.675 | 154ms |
| transistor1 | 0.875 | 0.487 | 0.571 | 142ms |
| regulator | 0.829 | 0.541 | 0.667 | 78ms |
| pcb(主板) | 0.825 | 0.329 | 0.438 | 167ms |
| usb | 0.825 | 0.493 | 0.633 | 178ms |
| usb_adaptor | 0.800 | **0.215** | **0.259** | 84ms |
| switch | 0.800 | 0.396 | 0.475 | 170ms |
| audiojack | 0.738 | 0.396 | 0.550 | 71ms |
| button_battery | 0.713 | 0.372 | 0.292 | 81ms |
| end_cap | **0.675** | 0.229 | 0.338 | 72ms |
| **均值(n=12)** | **0.817** | **0.412** | **0.509** | |

**与现有5类成绩单(0.920/0.511/0.582)的落差是真实的,不是退步**:5类里有hazelnut/
cable/pill这类相对容易的MVTec类目;Real-IAD全是真实产线电子件、缺陷极小(GT掩膜中位
只占图~0.4%)。**这批更难,但也更像赛题。**

### 弱项分两类,治法不同(重要)
**A. 图级判决弱(检测就漏了,定位再准也没用)**:end_cap 0.675、button_battery 0.713、
audiojack 0.738。杠杆在检测门(EAD/DINO融合、阈值标定),**不在分割头**。
注意audiojack:acc只有0.738但IoU 0.396/框命中0.550——检出的都定位得不错,是纯检测问题。

**B. 定位弱**:usb_adaptor 0.215/0.259(全场最差)、end_cap 0.229、pcb 0.329。缺陷太小。

**C. button_battery特殊**:IoU 0.372还行但框命中只有0.292——掩膜大致对但框对不上,
疑似掩膜碎裂或框合并距离(box_merge_d)标定不适配该类目。

**这个区分很重要**:此前所有精力都花在定位上,但这批数据显示至少3个类目的瓶颈在
**图级判决**。以前只有5个类目时看不到这个模式。

## 阈值标定缺陷 + 评测协议重做(2026-08-25)

### 发现:阈值被锚定在30张fit缺陷上,系统性漏检
`FewShotAdapter._calibrate` 的候选阈值**只来自观测到的分数**。两类分数之间有大空隙时,
空隙里一个候选都没有,阈值只能贴到缺陷侧端点=**30张fit缺陷里分数最低的那张**。
测试集里任何比"这30张中最弱的一张"还弱的缺陷,全部漏掉。

**真手机屏实测**(scripts/diag_phone_detect.py):正常分0~3.35、缺陷分1.7亿~6.3亿,
**两类零重叠**,阈值取3.35可零误报100%召回;而标定值是4.23亿 → **召回仅46%**。
缺陷越小漏得越狠(完全单调):最小25%档召回**4%**、25~50%档24%、50~75%档72%、最大25%档84%。

**错在锚定对象**:100张正常图有代表性,30张缺陷只是缺陷总体的小样本、必然不含最弱的那些。
锚在缺陷侧=系统性保证漏检。

### 无条件下移阈值:**判负**(Real-IAD 5类目实测)
改成 geomean(n_below, best_t) 无条件下移:
- 真手机屏:检出率 46%→**100%**、框命中 0.293→**0.533**、含漏检IoU 0.266→0.485 ✅
- Real-IAD 5类目均值:**图级acc -0.025、框命中 -0.044** ❌(误报代价)

我起初判断"分布重叠时不生效"是**错的**:n_below是"低于阈值的最高正常分",即使重叠
也常有间隙,于是几乎所有类目阈值都被下移、误报普遍上升。
→ 改用**保守版**:加倍数门槛 GAP_RATIO=10,只在空隙病态时才下移
(手机屏空隙1.26亿倍→触发;正常类目2~3倍→不触发,行为与原来完全一致)。验证中。

### 交付级bug:输出框的坐标系错了(**已修**)
`submit.py` 直接输出 `o["boxes"]`,而那是 **seg_eval_hw(256²) 掩膜坐标系**的框。
评委喂 2500² 原图,我们输出 256 空间的坐标 → **差9.77倍,定位分接近归零**。
**内部成绩单永远发现不了**:那边预测掩膜和GT掩膜都缩到256再比,口径自洽。
已加 `_scaled_boxes()`,用 PIL 只读文件头拿原图尺寸线性还原(不能用load_fast后的
张量,那个已被缩到长边1152)。验证:(10,20,30,40)→(98,195,293,391),比值9.77 ✓

### 评测协议重做:scripts/run_exam.py(一次fit + 混合流)
此前每个类目各跑一次fit(12次),而**赛场只迁移一次**。且每类只测40缺陷+40正常=80张,
赛题是1000+张——80张里一张翻转就是1.25%acc,而我们一直用0.01量级差异判断改进,那在噪声里。

新协议(严格按赛题):
- **一次fit**:100张2500²正常图 + 30张2500²缺陷图,从 phone_battery/sim_card_set/pcb
  三个**字面意义上的手机部件**均摊凑齐
- **同类拼接**:每张2500² = 同一产品4张图各放大到1250²摆2×2
- **混合测试流**:1000张,缺陷占30%(产线现实),打乱后逐张送入
- **固定取图不随机**,完全可复现
- 产出单文件离线HTML报告(fit阶段+测试阶段可视化,可按 全部/正确/漏检/误报 筛选)

### 数据能否撑起2500²完整协议
| 数据 | 正常板(要100) | 缺陷板(要30) |
|---|---|---|
| Real-IAD phone_battery | 最多174 ✅ | 75 ✅ |
| Real-IAD pcb | 139 ✅ | 111 ✅ |
| 手机屏MSD | **最多5** ❌(仅20张正常图) | 300 ✅ |
**只有Real-IAD能严格复刻2500²完整协议;手机屏正常图不足,只能验类型归属与定位。**

## 【测试脚本的两个致命bug】评测协议重做时踩的(2026-08-25 深夜)

### bug①:拼接放大毁掉精度测试(已修:精度只用原生分辨率)
把Real-IAD的256²图放大到1250²再拼2500²——**纯插值不增信息,只把缺陷糊开**,拼接后
缺陷占比又稀释4倍到~0.1%,图级分必然被淹没。同一批数据:
  **拼接放大版 acc=0.300** vs **原生单图版 acc=0.747**
**我据此得出的"混类会让图级判据崩掉"是错误归因** —— 混类其实扛得住(0.747)。
现在:`EXAM_STITCH=1` 只用于延时/形状压测,精度一律走原生(同铁律#4对AD2的处理)。

### bug②:"固定顺序取图"取成了字母序 → fit与test缺陷类型几乎不重叠(已修)
MVTec缺陷按类型分目录、目录名字母序。顺序取前40张的后果:
| 类目 | fit见到的 | test见到的 |
|---|---|---|
| cable | bent_wire/cable_swap/combined | **missing_cable/missing_wire/poke_insulation** |
| hazelnut | crack/cut/hole | hole/**print** |
| carpet | color/cut/hole | hole/**metal_contamination/thread** |
**等于拿A/B/C标阈值去测D/E/F**,完全解释了"fit重叠3/30、test重叠98.9%"这个矛盾。
赛题的30张缺陷是与测试集**同分布**采样的。修法:**固定种子打乱**(仍完全可复现,
只消掉字母序聚类)。修后fit能按比例见到全部缺陷类型。
**注:run_scorecard.py的prep_realiad本来就shuffle了,没这个bug;只有新写的模拟考脚本有。**

## 2500²赛场级现状(bug②修复前的数,仅供参考,修复版在跑)
| | 值 | 判断 |
|---|---|---|
| 延时 | 中位110ms / p90 129ms | ✅ 预算200ms,**赛场可用** |
| 框命中@0.5 | **0.552** | ✅ 与最好的成绩单持平(0.582) |
| 含漏检IoU | **0.511** | ✅ 等于现有基线 |
| 图级acc | 0.437(误报80%) | ❌ 受bug②污染,待复测 |
**定位与延时在2500²原生1024²拼接下是过关的**,窟窿只在图级判决。

## 阈值标定:"改成普通准确率"是个假修复(判负,别再试)
三份数据显示平衡准确率准则朝两个方向出错(手机屏阈值偏高漏检54%;混类/2500²偏低
误报25.7%/80.2%),我一度想改成普通准确率。**但普通准确率在两类严重重叠+正常图占
多数时会退化成"永远不报警"** ——合成验证:plain阈值9.90/acc0.762但**召回0%**。
而2500²那轮所谓"最优阈值acc=0.708",600张里缺陷180张,**全判正常就是0.700** ——
所谓+0.272提升是假象。**真正的问题是图级分本身没信号(重叠98.9%),不是阈值规则。**
已把默认改回平衡准确率;`CALIB=plain`保留为opt-in。

## 两个待验证的opt-in开关(均已实现,默认关,62个单元测试全过)
- `seg_gate=True`:**用分割图当图级判据**(取分割图前0.1%像素均值→z归一→标阈值)。
  依据:2500²上图级分重叠98.9%(没信号)但框命中0.552/IoU0.511(像素级有信号)。
  代价:正常图不能再走"判正常立即返回"的早退路径。
- `per_mode_gate=True`:**正常图按DINO的CLS聚成K个模态,每模态各自标定**。
  依据:AHL(赛题唯一点名文献)的第一步就是正常图聚类;混类fit时正常分被拉到
  1.7~6.0导致z归一失效。K自动定(降不到20%就不分),单一产品时K=1逐位退化。
