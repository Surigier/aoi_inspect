1. 全部用中文回复
2. 称呼规则：每次回复必须使用 leon 作为称呼
3. 遇到不确定的代码设计问题时，必须先询问 leon，不得直接行动
4. 代码兼容性：不能写兼容性代码，除非主动提要求
5. 代码风格：必须遵守 karpathy-guidelines 里的四条行为准则(Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution)
6.不要总是自大，总是不看别人的代码，不要总是自己猜


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

## 当前成绩(2026-07-20最终,统一口径load_fast+DINO永远融合+延时梯队重排,本机4060L)

| 类            | 图级acc   | 含漏检IoU | 纯定位IoU | 框命中@0.5 |
| ------------- | --------- | --------- | --------- | ---------- |
| hazelnut      | 0.882     | 0.629     | 0.719     | 0.625      |
| cable         | **0.927** | **0.811** | 0.811     | **0.933**  |
| pill          | 1.000     | 0.440     | 0.440     | 0.426      |
| pcb           | 0.825     | 0.253     | 0.260     | 0.287      |
| phone_battery | 0.875     | 0.370     | 0.376     | 0.475      |
| **均值**      | **0.902** | **0.501** | 0.521     | **0.549**  |

历史基线:含漏检IoU均值0.484/框0.600。**均值含漏检IoU=0.501首次突破0.5,超过历史
基线**。cable问题**已彻底根治并确定性复现**(见下),不再是开放问题。
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
