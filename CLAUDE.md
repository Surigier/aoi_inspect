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

## 当前成绩(2026-07-19最终,统一口径load_fast+DINO永远融合,本机4060L)
| 类 | 图级acc | 含漏检IoU | 纯定位IoU | 框命中@0.5 |
|---|---|---|---|---|
| hazelnut | 0.895 | 0.649 | 0.719 | 0.625 |
| cable | 0.255⚠️ | 0.747 | 0.811 | 0.867 |
| pill | 1.000 | 0.440 | 0.440 | 0.426 |
| pcb | 0.825 | 0.253 | 0.260 | 0.287 |
| phone_battery | 0.912 | 0.370 | 0.376 | 0.475 |
| **均值** | 0.777 | **0.492** | 0.521 | **0.536** |
历史基线:含漏检IoU均值0.484/框0.600。**均值含漏检IoU=0.492已超过历史基线**,框命中
0.536。定位(seg_head)确认恢复基线;DINO门改永远融合后全类无害(hazelnut/pill/pcb/
battery纹丝不动),cable从-0.64量级灾难收窄为单类拖累。

**cable未竟问题(⚠️开放,勿标记为已解决)**:该类EAD学生-教师原始异常分对"训练台架↔
测试台架"系统性差异极度敏感,且这种不稳定性来自**同进程内该类排第几个训练**(消耗的
随机数流位置不同→EAD学生具体权重不同,与代码/种子号无关)——已连续观测到3种不同
表现:①acc=0.236(几乎全判缺陷,640口径无融合门时)②acc=0.909(融合门"运气好",
诊断脚本独立跑cable第一时)③acc=0.255(融合阈值这次标偏敏感,正常图大量误报,统一
口径永远融合时)。永远融合已把"要不要用DINO"的赌博拿掉,但**阈值本身仍在不稳定的
EAD分布上标定**,这才是残留病根。下一步方向:让EAD学生训练/阈值标定本身更鲁棒
(如多种子平均阈值而非只平均检测分),而非继续在门控层面打补丁。
延时:最轻档p90≈205ms@4060L(compile_infer=True,悲观代理);2060真机待验。

## 重大负结果(勿重蹈,证据在commit与代码注释)
- DINO门"3折CV决定开关"已废弃(commit 4cdc115):cable上被证明是掷硬币而非真信号,
  fit侧对"test集系统性漂移"结构性看不见,同seg_head/component_graph今天的教训一致。
  改为默认永远融合(风险不对称:pcb过度触发代价仅-0.011,漏融合代价可达-0.6+)。
- **新seg_head(bagging+soft target+OOF三阈值)已回退**(commit 5f83c3b):8类实测3平5负
  (pcb 0.251→0.028崩塌),根因=OOF抛弃头阈值跨头迁移失败,绝对值/分位数两版迁移
  互斥失败。代码留在aoi/seg_head.py作opt-in研究件。
- crop_cascade(独立crop-head级联):ViSA pcb1实测-0.059,门控自动禁用。
- 按类选新旧seg_head的fit侧门控:CV原理上测不到fit/test漂移,3/3选错,已撤。
- 组件图"默认开":纹理类伪组件-0.218灾难;fit侧±0.1小信号边际增益估计=掷硬币。
- 更早:DINO/SubspaceAD定位、AnomalyCLIP融合、RAMS-R上生产、CutPaste合成、
  roi_zoom原版——全负,勿重开。
- GPU"脏卡"陷阱:连轴跑后SM降频,延时读数系统性偏高;测延时前查nvidia-smi温度/频率。

## 待办(优先级序,含文献路线图讨论共识2026-07-19)
1. **cable残留不稳定性**(见上"未竟问题"):EAD训练/阈值标定鲁棒性,不是门控层面的活
2. **UniVAD v2**(component_graph.py增量):补Hungarian匹配+关系图边(距离/包含/排列
   顺序),v1只有逐组件z-score、没有显式"缺了几个组件""顺序错了"的判断——真正对应
   缺件/错序的信号,值得投入
3. **DCP-SFR边界残差头**(opt-in,慎重):零初始化残差修正+OOF在raw/refined/SAM三者
   选,架构上几乎是RAMS-R(已因"8张留出门控噪声漏判强类"生产判负,见下)的v2——
   若做,门控必须上k折而非单次小留出,否则大概率重蹈覆辙
4. pcb/battery微小缺陷仍是主拉分点(0.26/0.37量级)
5. TF-IDG生成增广:官方代码github.com/rubymiaomiao/TF-IDG,需>8GB VRAM机器跑生成
   (AnyDoor ckpt+DINOv2 ViT-G),本机侧门控评审脚本scripts/run_tfidg_gate.py已就绪
   (3切分OOF均升+最差类回退≤0.01+真实占比≥50%三条全过才准入)
6. Boxes2Pixels(低优先级/待定):仅当官方30张缺陷标注只有框、没有精确掩膜时启用
   (SAM出伪掩膜训紧凑学生+单向自纠正);标注格式未知前不必先实现
7. 2060真机延时验证:scripts/run_2060_check.py(合成图,免数据集);4060L(256GB/s)
   悲观代理+2070(448GB/s)乐观代理可夹逼2060(336GB/s)
8. GitHub push(repo Surigier/aoi_inspect私有,token用时向用户要)

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
- 实验日志写_logs/(gitignore),用nohup跑长任务(会话重启不陪葬)
