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

## 当前成绩(2026-07-20最终,统一口径load_fast+DINO永远融合+延时梯队重排,本机4060L)
| 类 | 图级acc | 含漏检IoU | 纯定位IoU | 框命中@0.5 |
|---|---|---|---|---|
| hazelnut | 0.882 | 0.629 | 0.719 | 0.625 |
| cable | **0.927** | **0.811** | 0.811 | **0.933** |
| pill | 1.000 | 0.440 | 0.440 | 0.426 |
| pcb | 0.825 | 0.253 | 0.260 | 0.287 |
| phone_battery | 0.875 | 0.370 | 0.376 | 0.475 |
| **均值** | **0.902** | **0.501** | 0.521 | **0.549** |
历史基线:含漏检IoU均值0.484/框0.600。**均值含漏检IoU=0.501首次突破0.5,超过历史
基线**。cable问题**已彻底根治并确定性复现**(见下),不再是开放问题。
延时:最轻档p90≈205ms@4060L(compile_infer=True,悲观代理);2060真机待验。

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
  >=2/3类中位数为正且min(Δ)>=-0.01)双双不通过:LoRA_r2 median(Δ)=0.000
  (0/3类为正),LoRA_r4 median(Δ)=+0.001(2/3类为正,远低于0.005门槛)。封存前
  诊断排除"配置太保守/适配器没真动"的可能:保守配置下||ΔW||/||W_base||已达
  1.3~1.7e-3(远超1e-4死区);唯一压力测试(fruit_jelly r4/lr=1e-3/300步)证实
  权重可大幅移动(ΔW/W达4.5e-2)且fit/test同向大涨(+0.108/+0.114)。**准确结论:
  fruit_jelly上确实可能存在类别特定的真实收益,压力测试没有排除这一点——不能说
  "WRN表示已经足够"或"绝非超参数问题"。能确定的只是收益不广泛、不稳定(3类中
  2类稳定打平),继续在这个方向调参的竞赛期望值低。** 默认关,代码留opt-in研究件。
  最大价值:排除了一个"零时延但可能提分"的诱人方向,让资源能放心转向已有明显
  候选框信号的参考ROI。
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
- 更早:DINO/SubspaceAD定位、AnomalyCLIP融合、RAMS-R上生产、CutPaste合成、
  roi_zoom原版——全负,勿重开。
- GPU"脏卡"陷阱:连轴跑后SM降频,延时读数系统性偏高;测延时前查nvidia-smi温度/频率。

## 待办(优先级序,2026-07-24更新——WRN-LoRA已封存判负,Top-1参考ROI三路验证均未净正)
1. **数据缺口:找/补带normal模板的大分辨率同域(手机/电子件)数据**——Top-1参考ROI
   在Real-IAD pcb/phone_battery(256×256,无headroom)/MVTec LOCO(800~1700px但
   域不匹配,3/3类OOF全负)/pku_pcb(2000~3000px真PCB但无normal图)三条路线上都
   没能验证出净正(见"重大负结果"Top-1参考ROI条目),根因是本地没有同时满足"同域+
   大分辨率+有normal参考"的数据,不是机制本身被证伪。找到/补齐这类数据前,不建议
   继续在现有三条路线上调参。
2. **pcb/battery微小缺陷仍是主拉分点**(0.26/0.37量级)——DCP-SFR边界残差、UniVAD v2
   Hungarian、WRN-LoRA三条路线都已验证判负(见"重大负结果"),这几类目前没有已
   验证的改进候选在手,需要重新想机制而非在现有分割头上加小修正
3. TF-IDG生成增广:官方代码github.com/rubymiaomiao/TF-IDG,需>8GB VRAM机器跑生成
   (AnyDoor ckpt+DINOv2 ViT-G),本机侧门控评审脚本scripts/run_tfidg_gate.py已就绪
   (3切分OOF均升+最差类回退≤0.01+真实占比≥50%三条全过才准入)
4. Boxes2Pixels(低优先级/待定):仅当官方30张缺陷标注只有框、没有精确掩膜时启用
   (SAM出伪掩膜训紧凑学生+单向自纠正);标注格式未知前不必先实现
5. 2060真机延时验证:scripts/run_2060_check.py(合成图,免数据集);4060L(256GB/s)
   悲观代理+2070(448GB/s)乐观代理可夹逼2060(336GB/s)
6. GitHub push(repo Surigier/aoi_inspect私有,token用时向用户要)

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
