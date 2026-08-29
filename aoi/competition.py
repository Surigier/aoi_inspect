"""赛场大图统一检测器(CompetitionLargeDetector):2500² 输入的生产路径。
架构(2026-07 现状):
- 检测主干 = EfficientAD 整图卷积(学生-教师,无记忆库→延时恒定;阈值标在 EAD 分上)。
- 图级门 = DINOv2 受控 co-detector(3折CV验证,仅当融合门不劣于EAD-only才逐类启用;
  max(z_EAD,z_DINO) 治EAD漏检→提含漏检IoU;此前补检门/定位融合双死,平等融合是活路)。
- 定位 = WRN50 浅层(1,2)@512 监督分割头(30掩膜训练,双头logit集成 + SAM边界精化)。
- 辅助色彩/尺寸/结构分支只做缺陷【类型归属】(最强z分支=类型),不参与检测融合
  (少样本估融合权重不可靠,会拖垮强EAD;实测软融合在对抗数据反低于EAD单独,故弃)。
延时 ≈ EfficientAD(~140-170ms@2060)+ DINO门前向(~25ms)+ 轻分支(几ms),仍 <200ms。"""
import torch
import torch.nn.functional as F
from .fusion import znorm
from .fewshot import FewShotAdapter
from .backbone import Backbone
from .tiled_efficientad import TiledEfficientAD
from .branches.color_ad import ColorADBranch
from .branches.dimension_ad import DimensionADBranch
from .branches.structural_ad import StructuralADBranch
from .seg_head import map_to_boxes
# 生产seg_head=旧实现(双头集成+pooled-F1自洽阈值)。新实现(bagging+soft loss+OOF阈值,
# aoi/seg_head.py)8类实证:AD2三类净平、成绩单5类全输(均值-0.104,pcb 0.251→0.028/
# battery 0.374→0.122崩塌——OOF抛弃头阈值跨头迁移失败,绝对值/分位数两种迁移都不成:
# 分位数版救pcb却砸hazelnut -0.204)。唯一确证收益sheet_metal+0.111不抵。按纪律回退,
# 新头及OOF基建留作opt-in研究件(run_seg_head_ab_scorecard.py为证据)。
from ._seg_head_old_ae5fbbb import SupervisedSegHead
from .gcad_embed import fit_embed_ae, calibrate_zscore as _gcad_calibrate_zscore


def _mask_np(mk, hw):
    """(H,W){0,1} numpy → hw 大小(最近邻)。"""
    import numpy as np
    if mk.shape == tuple(hw):
        return mk
    t = torch.from_numpy(mk.astype("float32"))[None, None]
    return (F.interpolate(t, size=tuple(hw), mode="nearest")[0, 0].numpy() > 0.5).astype("uint8")


def _down(img, size):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


def _fit_modes(C, en, dn, kmax=4, seed=0):
    """把正常图的CLS向量聚成K个模态,每个模态存中心和该模态的(EAD/DINO)分数统计量。
    K的选法:从2到kmax试,取"模态内分数方差之和"下降最明显的那个;若K=1已经足够
    (下降不到20%),返回None表示不分模态——**单一产品时必须退回原行为**。"""
    import numpy as np
    X = torch.nn.functional.normalize(C.float(), dim=1)
    base = float(np.var(en) + np.var(dn))
    best = None
    for k in range(2, min(kmax, len(X) // 8) + 1):
        lab = _kmeans(X, k, seed)
        if min((lab == j).sum() for j in range(k)) < 5:      # 有簇太小 → 不可靠
            continue
        wv = sum(float(np.var(en[lab == j]) + np.var(dn[lab == j])) * (lab == j).sum()
                 for j in range(k)) / len(X)
        if best is None or wv < best[1]:
            best = (k, wv, lab)
    if best is None or best[1] > 0.8 * base:                 # 分模态没带来明显收益 → 不分
        return None
    k, _, lab = best
    modes = []
    for j in range(k):
        m = lab == j
        modes.append(dict(c=X[torch.from_numpy(m)].mean(0),
                          emu=float(en[m].mean()), esd=float(en[m].std() + 1e-9),
                          dmu=float(dn[m].mean()), dsd=float(dn[m].std() + 1e-9),
                          n=int(m.sum())))
    return modes


def _kmeans(X, k, seed=0, iters=30):
    import numpy as np
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:k]
    C = X[idx].clone()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = torch.cdist(X, C)
        new = d.argmin(1).numpy()
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[torch.from_numpy(lab == j)].mean(0)
    return lab


def _assign_mode(cls_vec, modes):
    """测试图归到最近的模态(用DINO的CLS,推理期零增量前向)。"""
    if len(modes) == 1 or modes[0]["c"] is None:
        return 0
    v = torch.nn.functional.normalize(cls_vec.float()[None], dim=1)[0]
    return int(max(range(len(modes)), key=lambda j: float(v @ modes[j]["c"])))


def _mode_z(e, d, m):
    return max((e - m["emu"]) / m["esd"], (d - m["dmu"]) / m["dsd"])


class _EADBranch:
    """EfficientAD 核心(整图卷积),映射到'外观缺陷'(一般异常的默认归类)。"""
    defect_type = "外观缺陷"

    def __init__(self, **kw):
        self.det = TiledEfficientAD(**kw)

    def fit(self, normals, defects, retrain_student=True):
        self.det.fit_fewshot(normals, defects, retrain_student=retrain_student)

    def score(self, img):
        return self.det._image_score(img)


class _AuxBranch:
    """轻量辅助分支(色彩/尺寸/结构),在下采样大图上跑,负责对应缺陷类型。"""
    def __init__(self, branch, defect_type, size):
        self.branch = branch
        self.defect_type = defect_type
        self.size = size

    def fit(self, normals, defects):
        self.branch.fit(torch.cat([_down(n, self.size) for n in normals], 0))

    def score(self, img, pre=None):
        """pre:预先算好的_down(img,size)张量(3路辅助分支size相同,判缺陷时共享一次下采样)。"""
        return self.branch.infer(pre if pre is not None else _down(img, self.size)).score


class CompetitionLargeDetector:
    def __init__(self, device="cuda", aux_size=320, train_steps=10000, seg_eval_hw=(256, 256),
                 compile_infer=False, sam_refine=True, roi_zoom=False, seg_in=512, rams=False,
                 ead_students=2, crop_cascade=False, comp_graph=False, boundary_refine=False,
                 tta=False, gcad_embed=False, joint_ensemble=True, use_vlm_type=True, per_mode_gate=False, seg_gate=False, dino_seg=False):
        # ead_students=2:多种子EAD学生集成(检测分,教师共享)。实测工作类方差收窄~2×、
        # 现场一次fit下限+0.012、均值零代价;fit不计时训练免费,推理+一次学生前向(延时待验)。
        # rams 默认关:RAMS-R残差注意力修正支隔离探针+0.013~0.033(3/4类),但生产全管线实测净平
        # (SAM下游重塑边界冲掉raw增益+8张留出门控噪声漏判强类+框对掩膜形变敏感),且开门类延时
        # 超线(295/330ms,修正支重复提特征)。留opt-in备查,见 run_rams_diag.py / seg_head._RamsCorr。
        # roi_zoom 默认关:AD2真大图实测负面(0.131→0.072,裁块尺度失配+阈值不匹配),留待修复
        dev = device if torch.cuda.is_available() else "cpu"
        bb = Backbone(device=dev)
        self.branches = [
            _EADBranch(device=dev, train_steps=train_steps, compile_infer=compile_infer,
                       n_students=ead_students),
            _AuxBranch(ColorADBranch(grid_size=16), "色彩变化", aux_size),
            _AuxBranch(DimensionADBranch(), "尺寸偏差", aux_size),
            _AuxBranch(StructuralADBranch(backbone=bb, grid_size=16), "缺件/逻辑", aux_size),
        ]
        self.aux_size = aux_size
        self.use_vlm_type = use_vlm_type
        self._dino_feat = None               # dino_seg专用的独立DINO提取器(见_wrn_dino_feats)
        self._wrn_cache_on = False; self._wrn_cache = None   # 见 _wrn_feats:作用域限定在locate内
        # 操作员反馈样本(见 _calibrate_dino_gate 末尾的硬约束):这些图是**人亲口标的**,
        # 不能和另外130张fit样本平权投票,否则1票在130票里等于没投。
        self._fb_defects = []; self._fb_normals = []; self._fb_unsat = None
        self.dino_seg = dino_seg             # 分割头吃WRN⊕DINO拼接特征(见_wrn_dino_feats),默认关
        self.seg_gate = seg_gate             # 用分割图当图级判据(见_calibrate_seg_gate),默认关
        self.per_mode_gate = per_mode_gate   # 正常图分模态标定阈值(见_calibrate_dino_gate),默认关,验证后再开
        self.type_head = None       # VLM监督的类型归属头,fit期建;不可用则保持None走_ztype
        self.stats = []
        self.weights = []
        self.threshold = None
        # 定位专用浅层骨干:WRN50 layers(1,2) @512 → 128²特征格(现状40²比微小缺陷粗)。
        # 扫描实测(run_feat_res.py):IoU均值0.305→0.449(+47%),pcb+53%/battery+74%/pill+72%,
        # 且浅层更快(8ms vs 36ms)。640过犹不及。结构分支仍用默认bb(layers 2,3)不受影响。
        self._bb_loc = Backbone(layers=(1, 2), device=dev)
        self._seg_in = seg_in                              # 定位特征输入分辨率(大图可调高保小缺陷)
        self._bb_l3 = None                                 # layer3 惰性建(RAMS-R修正支多尺度用)
        # joint_ensemble默认开(2026-08-17验证):分割头的线性头+卷积头**联合训练**
        # (梯度同时流过两个头让它们协同分工),而不是各自独立训300步再拼成_Ensemble。
        # 6类验证ΔIoU=[+0.006,+0.003,+0.001,-0.010(pcb),+0.015,+0.043(breakfast_box逻辑
        # 异常)],median=+0.004 mean=+0.010 min=-0.010,框命中均值+0.018,**图级acc
        # 6/6类完全不变**(不碰图级判定,没有GCAD那种假阳性风险)。严格margin判据差
        # 一点(median+0.004 vs 0.005线),唯一负例是pcb(微小缺陷,赛题里出现概率低,
        # 用户已明确降优先级);排除pcb后5类median=+0.006/min=+0.001全正。**零成本**:
        # 推理结构完全不变(还是同一个_Ensemble)、零延时增量、不碰骨干。
        # 验证脚本seghead_tuning/probe_joint_ensemble.py。
        self.seg_head = SupervisedSegHead(device=dev,
                                          extractor=(self._wrn_dino_feats if dino_seg else self._wrn_feats),
                                          rams_extractor=self._rams_scales if rams else None,
                                          joint_ensemble=joint_ensemble)
        self.seg_eval_hw = seg_eval_hw
        self.pix_thr = None                                # 像素图二值阈值(正常分位标定)
        from .sam_refine import SamRefiner
        self.sam = SamRefiner() if sam_refine else None    # SAM边界精化(仅判缺陷图触发)
        self.roi_zoom = roi_zoom                           # 2500²大图:粗定位→原生裁块→局部重分割
        # crop_cascade默认关:独立crop-head级联(重做roi_zoom,候选来自ECC模板残差非粗分割头,
        # 独立mu/sd/阈值),fit时OOF留出验证净正才启用,opt-in等待真实数据确认前不默认开。
        self.use_crop_cascade = crop_cascade
        self.crop_cascade = None
        # comp_graph默认关:组件图逻辑异常分支(UniVAD思想轻量落地,SAM只在fit期打组件
        # 伪标签,热路径ECC+复用WRN特征ROI池化)。fit时OOF留出验证净正才启用,零回退。
        self.use_comp_graph = comp_graph
        self.comp_graph = None
        # boundary_refine默认关:DCP-SFR启发的边界残差头(浅层edge cue修正分割logit边界,
        # 零初始化残差)。fit时k折OOF验证净正才启用,零回退。
        self.use_boundary_refine = boundary_refine
        self.boundary_refiner = None
        # tta默认关(2026-07-20真实数据判负,run_tta_ab.py):5类均值Δ纯定位=-0.080/
        # Δ框=-0.060,3/5类明显负(sheet_metal-0.173/walnuts-0.205/fruit_jelly-0.033),
        # 2/5类接近持平(pcb+0.011/battery+0.001),没有一类真正获益。当初"确定性技术,
        # 不依赖fit侧判断,风险性质与已判负的学习型小修正不同"的推理本身没错,但漏了
        # 一个新失败模式:seg_head/WRN特征和标定统计量本来就不是为翻转不变性设计的,
        # 喂模型从没见过的镜像图产出的不是"降噪的另一视角"而是系统性更差的预测,平均
        # 进去反而拖累整体。opt-in代码留档,不进生产候选。
        self.use_tta = tta
        # gcad_embed默认关(2026-08-13已回退):DINOv2 CLS token瓶颈自编码器判整图语义
        # 构图,补EAD/DINO patch级比对看不到的逻辑异常。当天研究阶段的验证(global_context/
        # eval_aggressive.py、eval_emb_prod5.py)只在test_defs(缺陷图)上测IoU/hit,
        # **从没测过正常图会不会被误报**,得出"min=0.000零回退"的结论有方法论盲区。真上
        # 生产scripts/run_scorecard.py(含正常图的完整acc)一测:图级acc从0.902崩到0.703
        # (-0.199),IoU/框命中只有+0.006~+0.022的小幅提升,完全不能抵消——OR门只要
        # 独立阈值稍微松一点,在"正常图占多数"的真实测试集上误报绝对数量就会很大,不是
        # "min=0.000"这种只看缺陷图的口径能测出来的。按零回退纪律立刻回退,默认关。
        # 代码留opt-in研究件(aoi/gcad_embed.py),真正要用必须先补上正常图假阳性率的
        # OOF验证,不能只测缺陷图召回这一侧。
        self.gcad_embed = gcad_embed
        self._embed_ae = None
        self._embed_stats = None
        self._embed_thr = None

    @torch.no_grad()
    def _wrn_feats(self, img):
        """img(3,H,W)[0,1] → WRN50 浅层(1,2)特征 (C,128,128)。
        先搬 GPU 再下采样(大图 CPU interpolate 慢),再提特征(~8ms)。

        **缓存有严格作用域:只在 locate() 内生效**(`_wrn_cache_on` 在 locate 入口开、
        finally 里关)。逐段计时实测:分割头 39.7ms、类型头 87.3ms,而**类型头里那次
        WRN前向与分割头刚算过的是同一份** —— 同一张图的骨干特征被算了两次。

        历史教训(别再整个删掉):早先加过"只在locate入口清空"的缓存,结果
        seg_head.fit() 在**fit期**逐张调用本函数,而fit期不进locate()、缓存永不清空,
        30张缺陷图全部拿到第一张的特征却配各自不同的掩膜去训练,定位直接崩
        (phone_battery 含漏检IoU 0.399→0.033、框命中0.550→0.013)。
        **当时的修法是整个删掉缓存,正确的修法是给它加作用域**——fit期开关为False,
        自然不命中;推理期同一张图内命中,省掉一次完整WRN前向。"""
        # 缓存**必须按张量身份命中**,不能盲取。血的教训:曾写成"开关打开就返回缓存",
        # 而 _wrn_feats_diff 会在同一次 locate 内先后传入**测试图和参考图**——参考图
        # 命中了测试图的缓存 → f - fr = f - f = 全零 → 分割头输入全是零 →
        # 定位归零(实测 框命中0.0000/IoU 0.0006)。
        # 用 (data_ptr, shape) 做键:同一次调用里两张图同时存活,地址必不相同,不会误命中。
        key = (img.data_ptr(), tuple(img.shape))
        if getattr(self, "_wrn_cache_on", False) and self._wrn_cache is not None \
                and self._wrn_cache[0] == key:
            return self._wrn_cache[1]
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = x.to(self._bb_loc.device)
        x = F.interpolate(x, size=(self._seg_in, self._seg_in), mode="bilinear", align_corners=False)
        out = self._bb_loc.extract(x)[0]
        if getattr(self, "_wrn_cache_on", False):
            self._wrn_cache = (key, out)
        return out

    @torch.no_grad()
    def _wrn_dino_feats(self, img):
        """WRN浅层(1,2) ⊕ DINOv2 patch特征,拼成分割头的输入。

        动机(文献+我们自己的数据同时指向):
          - Dinomaly(CVPR2025)在多类别异常检测上把DINOv2做成SOTA,结论是基础模型
            特征才是多类别鲁棒性的关键;
          - 我们自己的标定数据也这么说——三产品混合迁移时,EAD/WRN侧的可分性
            best_bal=0.638,而DINO侧是0.897。
        而分割头一直只吃WRN浅层。DINO的patch特征在图级门那步**本来就算了**
        (dino_gate._patches),只是没缓存;这里复用缓存 → **推理零增量前向**。

        DINO是518²输入、patch 14 → 37²网格;上采样到WRN的128²格后按通道拼接。
        DINO门未启用时静默退回纯WRN(不改变原行为)。"""
        f = self._wrn_feats(img)                              # (768,128,128)
        # **必须用独立的特征提取器,不能依赖 self._dino**:
        # fit_fewshot 里 seg_head.fit() 发生在 _calibrate_dino_gate() **之前**,
        # 那时 self._dino 还是 None → 训练时只拿到768通道,推理时却是1152 →
        # 和 fit 期存下的 mu/sd 对不上,直接 RuntimeError(1152 vs 768)。
        # 独立实例保证 fit/推理两边通道数一致。DINO门存在时优先复用它已缓存的
        # patch(零增量前向);不存在时用自己的实例算。
        g = getattr(self, "_dino", None)
        if g is None:
            if getattr(self, "_dino_feat", None) is None:
                from .dino_gate import DinoGate
                self._dino_feat = DinoGate(device=self._bb_loc.device)
            g = self._dino_feat
        # **按图身份校验**,不能只看"有没有缓存"——否则会拿到上一张图的patch。
        # (与WRN特征缓存同一类错误:第一版按状态盲取,导致参考图命中测试图的缓存、
        #  f-fr=全零、定位归零。缓存必须按输入身份命中。)
        key = (img.data_ptr(), tuple(img.shape)) if hasattr(img, "data_ptr") else None
        pg = getattr(g, "last_patch_grid", None)
        if pg is None or getattr(g, "last_key", None) != key:
            g._patches(img)                                   # 本图没跑过DINO(或缓存是别的图)
            pg = getattr(g, "last_patch_grid", None)
            if pg is None:
                return f
        d = F.interpolate(pg[None].to(f.device), size=f.shape[-2:],
                          mode="bilinear", align_corners=False)[0]
        return torch.cat([f, d], dim=0)

    @torch.no_grad()
    def _rams_scales(self, img):
        """RAMS-R 修正支多尺度特征:WRN layers(1,2)按通道拆两尺度 + 惰性layer3,统一128²格。
        与基线extractor解耦(tmpl_diff模式下修正支仍用本尺度组,additive不冲突)。"""
        f12 = self._wrn_feats(img)                            # (768,128,128)
        if self._bb_l3 is None:
            self._bb_l3 = Backbone(layers=(3,), device=self._bb_loc.device)
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(self._bb_loc.device)
        x = F.interpolate(x, size=(self._seg_in, self._seg_in), mode="bilinear", align_corners=False)
        f3 = self._bb_l3.extract(x)                           # (1,1024,g3,g3)
        f3 = F.interpolate(f3, size=f12.shape[-2:], mode="bilinear", align_corners=False)[0]
        s1, s2 = torch.split(f12, (256, 512), dim=0)
        return [s1, s2, f3]

    def _oof_aux_normal_scores(self, branch, normals, defects):
        """辅助分支(色彩/尺寸/结构)在正常图上的分布统计,必须用**留出法**算。

        原因(2026-08-21查实):这三个分支都是记忆库结构——fit(normals)把正常图特征
        存进bank,infer算到bank的最近邻距离。原来的写法是bank用normals建、又拿同一批
        normals去打分,**正常图在bank里匹配到自己,距离恒为0**:色彩分支实测
        mean=0/std=0(2维色度数值干净,精确为0),结构分支实测mean=1.97/std=0.35——
        那不是真实分布,是torch.cdist用矩阵乘法算欧氏距离的数值误差(1536维深度特征
        数值大,误差显著)。拿这种假统计量做z归一,跨分支完全不可比:结构分支z爆到
        197、其他分支只有1~7,**类型归属100%误判**(实测hazelnut的crack/cut/hole和
        pill的color共36张缺陷图,全部判成"缺件/逻辑",一张都没对)。

        2折留出:A半建库打B半、B半建库打A半,收集全部留出分——每张正常图的分都来自
        "库里没有它自己"的状态,才是真实的正常分布。统计量算完由调用方用全量normals
        重建生产库,推理质量不受影响。辅助分支很轻(320²、无训练),两次重建成本可忽略。"""
        n = len(normals)
        if n < 4:
            branch.fit(normals, defects)
            return [branch.score(x) for x in normals]     # 样本太少切不动,退回原行为
        half = n // 2
        A, B = normals[:half], normals[half:]
        branch.fit(A, defects)
        s_b = [branch.score(x) for x in B]
        branch.fit(B, defects)
        s_a = [branch.score(x) for x in A]
        return s_a + s_b

    def fit_fewshot(self, normals, defects, defect_masks=None, retrain_ead=True):
        """检测由 EAD 核心(branches[0])单独负责;辅助分支只为'类型归属'拟合并估 μ/σ。
        实测从少样本估融合权重不可靠(弱分支overfit→拖垮强EAD),故不做检测层融合。
        defect_masks:每张缺陷的 (H,W){0,1} 掩膜(赛题迁移图带标注)→ 训监督分割头提定位精度。

        retrain_ead=False:跳过EAD学生重训(其余标定全部照跑)。**仅当没有新增正常图时
        才允许**——EAD学生只在正常图上训,缺陷图只参与阈值标定,所以操作员反馈漏检
        (新增缺陷图)时重训学生纯属浪费。用于ActiveLearningLoop的增量反馈,见
        aoi/active_learning.py。"""
        self.stats = []
        for i, b in enumerate(self.branches):
            if i == 0:
                b.fit(normals, defects, retrain_student=retrain_ead)
                ns = [b.score(x) for x in normals]        # EAD是训出来的学生,不是记忆库,无自匹配问题
            else:
                ns = self._oof_aux_normal_scores(b, normals, defects)
                b.fit(normals, defects)                   # 生产库用全量normals重建(留出只为算统计量)
            m = sum(ns) / len(ns)
            s = (sum((x - m) ** 2 for x in ns) / len(ns)) ** 0.5
            self.stats.append((m, s))
        ead = self.branches[0]
        ns = [ead.score(x) for x in normals]
        ds = [ead.score(d) for d in defects]
        self.threshold = FewShotAdapter._calibrate(ns, ds)    # 阈值标在 EAD 分上
        if defect_masks is not None:
            d_imgs, d_masks = list(defects), list(defect_masks)
            if self.roi_zoom:
                ci, cm = self._native_crops(defects, defect_masks)   # 原生裁块样本(教头认放大视角)
                d_imgs += ci; d_masks += cm
            self._select_feat_mode(defects, defect_masks, normals)   # 留出集自动决定模板差分开关
            self.seg_head.fit(self._eff(), d_imgs, d_masks, normals[:30])
            self._calibrate_boxes(defects, defect_masks)             # fit标定碎框合并距离
        self._calibrate_pixel(normals)
        # VLM监督的类型归属头:fit期(不计时)用VLM给缺陷图打5类标签,蒸馏成质心分类器。
        # 推理期零API/零外网。VLM不可用时fit()返回False,type_head保持None,自动走_ztype启发式。
        self.type_head = None
        # 注意:即使VLM不可用,类型头也会以**规则模式**建立(位置匹配特征argmax,离线58%),
        # 优于退回_ztype启发式(38%)。评委机器无外网是工业评测常态,这条路径必须留着。
        if defect_masks is not None and self.use_vlm_type:
            from .type_head import VLMTypeHead
            th = VLMTypeHead()
            if th.fit(self, normals, list(defects), list(defect_masks)):
                self.type_head = th
        self.boundary_refiner = None
        if defect_masks is not None and self.use_boundary_refine:
            from .boundary_refine import BoundaryRefiner
            br = BoundaryRefiner(device=self._bb_loc.device)
            br.fit(self, defects, defect_masks)               # k折OOF验证净正才启用
            if br.enabled:
                self.boundary_refiner = br
        if defect_masks is not None and self.sam is not None:
            self.sam.calibrate(self, defects, defect_masks, seg_map_fn=self.segment)  # SAM受控精化OOF标定
            # (若boundary_refiner已启用,self.segment()此时已含边界修正,SAM门控看到的
            #  就是生产真实会用的base——两套门控解耦生效,不重复实现SAM那套逻辑)
        self.crop_cascade = None
        if defect_masks is not None and self.use_crop_cascade:
            from .crop_cascade import CropHeadCascade
            cc = CropHeadCascade(device=self._bb_loc.device)
            cc.fit(self, self._ref_bank, defects, defect_masks, normals)   # OOF留出验证净正才启用
            if cc.enabled:
                self.crop_cascade = cc
        self.comp_graph = None
        if self.use_comp_graph:
            from .component_graph import ComponentGraph
            cg = ComponentGraph(device=self._bb_loc.device)
            cg.fit(self, normals, defect_imgs=defects if defect_masks is not None else None,
                   defect_masks=defect_masks)               # OOF留出验证净正才启用(零回退)
            if cg.enabled:
                self.comp_graph = cg
        self.rescue_gray = None; self.rescue_seg_thr = None          # 救援默认关(v1-v6净收益≈0已放弃)
        if getattr(self, "use_rescue", False):
            self._calibrate_rescue(normals, defects)
        self._dino = None                                            # DINOv2 图级co-detector(受控)
        if getattr(self, "use_dino_gate", True) and defects:
            self._calibrate_dino_gate(normals, defects)
        self._seg_gate = None
        if self.seg_gate and defect_masks is not None and self.seg_head.head is not None:
            self._calibrate_seg_gate(normals, defects)
        self._embed_ae = None
        if self.gcad_embed and self._dino is not None:
            self._calibrate_gcad_embed(normals, defects)
        self._calibrate_latency(normals)                             # 延时预算自适应(评委真机自测自裁)
        return self.threshold

    def _calibrate_latency(self, normals):
        """延时预算自适应:fit发生在评委真机上且不计时→免费自测。探针必须是**真实原生分辨率
        文件**,禁止用已缩放张量(submit.py用load_fast(max_size=1152)加载normals/defects,
        若拿这些张量重建"probe文件"→ probe长边最多1152,原生2500²的真实解码耗时被系统性
        低估,自裁会因此偏松、真机可能超线)。优先用 probe_paths(submit传入的真实fit文件
        路径,同产线同分辨率的原生文件)直接端到端计时;没有时(如合成数据测试)才退化为
        重建模式并打印警告。
        端到端计时 load_fast解码 + locate最坏链(强制全判缺陷→SAM/框必走)。分解式预算(GPU链
        +解码相加)实测系统性低估30-40ms(张量上传/预处理隐性成本),故必须量真口径。
        超预算裁剪顺序(2026-07-19重排,按当前证据"每ms换的分"排序,不是历史顺序):
        ①第二EAD学生 ②EAD面积降档(max_pixels,Pareto扫描证实对纯定位IoU零影响,纯白拿)
        ③SAM(sam_refine.py逐区域OOF门控5/5域实测全判reject_all,已短路近零成本;若未
        标定/仍在跑推理则一并弃,反正当前无证实正贡献)④DINO门(cable实锤:融合门是
        cable唯一救命机制,弃它代价可达-0.6+量级,历史"SAM最值钱"的旧排序已过时,DINO
        才是最后才能动的)⑤最深max_pixels(700k,仍超硬线190时)。"""
        import time
        import tempfile
        from pathlib import Path as _P
        import numpy as np
        from PIL import Image as _Im
        from .imageio import load_fast
        budget = getattr(self, "latency_budget_ms", 170)
        if not budget or not normals:
            return
        hard = 190.0                                        # 端到端硬线(200留10ms余量)
        budget = min(budget, hard)

        probe_paths = getattr(self, "probe_paths", None)
        pf = None
        if probe_paths:
            def _native_size(p):
                try:
                    with _Im.open(p) as im:
                        return im.size[0] * im.size[1]
                except Exception:
                    return -1
            existing = [p for p in probe_paths if _P(p).exists()]
            if existing:
                pf = max(existing, key=_native_size)         # 原生分辨率最大的真实文件,不重建
        if pf is None:
            print("!! _calibrate_latency: 无probe_paths(真实文件路径),退化为张量重建探针"
                  "(可能低估解码耗时,submit.py应传入真实fit文件路径)", flush=True)
            probe = max(normals, key=lambda im: int(im.shape[-1]) * int(im.shape[-2]))
            fmt = str(getattr(self, "probe_format", "png")).lower().lstrip(".")
            arr = (probe.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            pf = _P(tempfile.mkdtemp(prefix="latprobe_")) / f"probe.{fmt}"
            if fmt in ("jpg", "jpeg"):
                _Im.fromarray(arr).save(str(pf), quality=92)
            else:
                _Im.fromarray(arr).save(str(pf))
        ead = self.branches[0].det.det                      # TiledEfficientAD→内层EfficientADDetector

        def timed():
            thr, dthr = self.threshold, getattr(self, "_dino_thr", None)
            self.threshold = -1e9                                    # 强制最坏链
            if getattr(self, "_dino", None) is not None:
                self._dino_thr = -1e9
            for _ in range(2):
                self.locate(load_fast(str(pf)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(4):
                self.locate(load_fast(str(pf)))              # 端到端=解码+预处理+locate(评分口径)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 4 * 1000
            self.threshold = thr
            if dthr is not None:
                self._dino_thr = dthr
            return ms

        def probe():
            """**取3次中位数**,不用单次。单次探针受GPU热状态/瞬时负载影响很大,
            而裁剪是个一次性的、不可逆的结构决策——用噪声测量去做这种决策,
            会让同样的代码和数据在不同时刻得出完全不同的模型。
            三种子实测:图级acc 在 0.669~0.913 之间摆动,正是这个机制在跳。"""
            import statistics
            return statistics.median([timed() for _ in range(3)])

        self.lat_trimmed = []
        self.lat_probe_ms = probe()
        if self.lat_probe_ms > budget and getattr(ead, "pairs", None) and len(ead.pairs) > 1:
            ead.pairs = ead.pairs[:1]                                # ①弃第二学生
            self.lat_trimmed.append("student2")
            self.lat_probe_ms = probe()
        # ②EAD面积预算降档(方形2500²的主开销;image-area是EAD唯一随图涨的GPU成本,
        #   WRN/SAM/DINO均定尺寸)。1.4M→1.1M→0.9M,每档~-20%EAD耗时,Pareto扫描证实
        #   对纯定位IoU零影响——白拿,排在DINO之前砍。
        tiled = self.branches[0].det
        for mp in (1_100_000, 900_000):
            if self.lat_probe_ms <= budget:
                break
            if getattr(tiled, "max_pixels", 0) > mp:
                tiled.max_pixels = mp
                self.lat_trimmed.append(f"max_pixels={mp//1000}k")
                self.lat_probe_ms = probe()
        # ③【SAM已移出可裁剪清单,勿再加回】逐段计时实测 SAM精化中位 **0.1ms**
        # (逐区域OOF门控已短路),砍它**省不到任何时间**;而没有SAM收紧掩膜,
        # 掩膜面积变大 → 类型头的参考特征池化开销随面积上涨(实测17.8ms→89.4ms)。
        # 砍SAM不但没省时间,反而让下游更慢——纯亏。
        # ④【DINO门已移出可裁剪清单,勿再加回】
        # 原逻辑:超真硬线时弃DINO门。但这笔交易**永远不划算**——弃它省的延时很少,
        # 精度代价却是-0.6量级(cable实锤 0.811→0.171)。而裁剪决策本身建立在一次
        # 噪声探针上,等于"用一次不可靠的测量,去赌掉整个图级判决能力"。
        # 实测后果:三种子下图级acc 0.669~0.913 剧烈摆动,而延时预算其实一直很宽裕
        # (中位82ms / 预算200ms)。宁可延时略超(检测时间按比例扣分),
        # 也不该让准确率崩掉。
        if self.lat_probe_ms > hard and getattr(tiled, "max_pixels", 0) > 700_000:
            tiled.max_pixels = 700_000                               # ⑤最深档(仍超真硬线时)
            self.lat_trimmed.append("max_pixels=700k")
            self.lat_probe_ms = probe()

    def _calibrate_rescue(self, normals, defects):
        """受控补检标定(榨干30张图级标签):EAD判正常时,监督信号超'fit零误翻线'才翻正。
        非对称救援≠对称融合(后者曾拖垮EAD已弃):只救漏检、fit正常集上保证零误翻。
        信号①seg头图max(30掩膜监督,最强);②辅助分支z分(色彩/尺寸/结构,显式缺陷型)。"""
        import numpy as np
        self.rescue_seg_thr = None
        self.rescue_aux_thr = None
        self.rescue_gray = None
        # v6(完全体):在B半上验证"联合翻正条件"(EAD灰区 ∧ 信号超线),而非只验线——
        # 实际翻正是联合事件,单验线既过严(hazelnut被误禁)又不够(pcb灰区内穿线没查)。
        # 灰区下界g自适应:取B半零误翻的最松g∈{0.95,0.9,0.85,0.8};0.95都守不住→禁用。
        half = len(normals) // 2
        A, B = normals[:half], normals[half:]
        if self.seg_head.head is None or not A or not B or self.threshold is None:
            return
        ead = self.branches[0]
        a_sig = np.array([float(self.segment(n).max()) for n in A])
        b_sig = np.array([float(self.segment(n).max()) for n in B])
        b_ead = np.array([ead.score(n) for n in B])
        d_sig = np.array([float(self.segment(d).max()) for d in defects])
        d_ead = np.array([ead.score(d) for d in defects])
        bar = float(a_sig.max() + (a_sig.max() - np.percentile(a_sig, 90)) + 1e-6)
        for g in [0.8, 0.85, 0.9, 0.95]:                     # 从松到紧,取第一个B半零误翻的
            b_flip = ((b_ead >= g * self.threshold) & (b_ead < self.threshold) & (b_sig >= bar)).sum()
            if b_flip == 0:
                # 证据量:该(g,bar)下能救到的fit缺陷数(漏检段:EAD分<阈值)
                d_flip = ((d_ead >= g * self.threshold) & (d_ead < self.threshold) & (d_sig >= bar)).sum()
                if d_flip >= 3:
                    self.rescue_seg_thr = bar
                    self.rescue_gray = g
                break                                        # g再收紧只会更少救,B已零误翻即停

    def _calibrate_dino_gate(self, normals, defects):
        """DINOv2 图级 co-detector 标定(受控平等融合,治EAD漏检→提含漏检IoU)。
        max(z_EAD, z_DINO) 联合门,默认永远启用(不再由fit侧3折CV决定开关)。
        改动原因(2026-07-19,cable实锤):EAD原始分在某些类上test集系统性失灵是
        fit阶段结构性看不见的(fit侧正常/缺陷分离良好,test分布漂移只在test暴露)——
        cable@生产配置实测:同一份代码/同一个种子号,仅因为该类在同进程里排第几个
        训练(消耗的随机数流位置不同→EAD学生具体权重不同),acc在0.909(融合门"运气好"
        判定开)和0.691(判定关,含漏检暴跌到0.048)间跳变。3折CV在这类问题上只是
        掷硬币,不是真的在测信号,margin放宽(bf>=be-0.03)只改了赢面没拿掉赌博本身。
        风险不对称:历史记录里DINO过度触发的已知代价仅pcb -0.011(小,单类单次观测);
        DINO缺失的代价是cable那种-0.6+量级灾难。故不再赌,默认永远融合;仅当DinoGate
        本身构建失败(异常/样本太少)才回退纯EAD。"""
        import numpy as np
        from .dino_gate import DinoGate
        self._dino = None
        if self.threshold is None or len(normals) < 9 or len(defects) < 6:
            return
        ead = self.branches[0]
        gate = DinoGate(device=self._bb_loc.device)
        gate.build(normals[:40])
        en, dn, cn = [], [], []
        for n in normals:
            en.append(ead.score(n)); dn.append(gate.score(n))
            cn.append(gate.last_cls.detach().cpu())        # score()顺手缓存的CLS,零增量前向
        ed, dd, cd = [], [], []
        for d in defects:
            ed.append(ead.score(d)); dd.append(gate.score(d))
            cd.append(gate.last_cls.detach().cpu())
        en = np.array(en); dn = np.array(dn); ed = np.array(ed); dd = np.array(dd)

        # ---- 正常样本的模态划分(per-mode 阈值)----
        # 为什么需要:此前全体正常图共用一套(均值,方差)。当那100张正常图里混了多个
        # 产品/型号/批次时,"正常"的分数范围被拉得极宽,z归一化就失去意义——实测混三个
        # 手机部件类目做fit时,**30张缺陷的分数全部落在正常区间内**(重叠30/30),
        # 平衡准确率只有0.64,图级判据几乎失效。
        # 做法:用DINO的CLS把正常图聚成K个模态,每个模态各自统计(均值,方差);推理时
        # 测试图先归到最近的模态,再用**那个模态**的统计量做z归一化。
        # **退化干净**:正常图本来就是单一产品时K=1,行为与改动前逐位一致。
        self._modes = None
        if self.per_mode_gate and len(normals) >= 30:
            self._modes = _fit_modes(torch.stack(cn), en, dn)
        if self._modes is not None:
            k_n = [_assign_mode(c, self._modes) for c in cn]
            k_d = [_assign_mode(c, self._modes) for c in cd]
            zn = [float(_mode_z(en[i], dn[i], self._modes[k_n[i]])) for i in range(len(en))]
            zd = [float(_mode_z(ed[i], dd[i], self._modes[k_d[i]])) for i in range(len(ed))]
            print(f"!! per-mode门: 正常图聚出{len(self._modes)}个模态 "
                  f"{[m['n'] for m in self._modes]}", flush=True)
        else:
            emu, esd = en.mean(), en.std() + 1e-9
            dmu, dsd = dn.mean(), dn.std() + 1e-9
            self._modes = [dict(c=None, emu=emu, esd=esd, dmu=dmu, dsd=dsd, n=len(en))]
            zn = [float(max((e - emu) / esd, (d - dmu) / dsd)) for e, d in zip(en, dn)]
            zd = [float(max((e - emu) / esd, (d - dmu) / dsd)) for e, d in zip(ed, dd)]
        m0 = self._modes[0]
        emu, esd, dmu, dsd = m0["emu"], m0["esd"], m0["dmu"], m0["dsd"]
        fz2 = lambda e, d: max((e - emu) / esd, (d - dmu) / dsd)   # 兼容explain()等只取单套统计量的调用
        self._dino = gate
        self._dino_stats = (emu, esd, dmu, dsd)
        self._dino_fuse = fz2
        self._dino_thr = FewShotAdapter._calibrate(zn, zd)
        self._apply_feedback_constraint(ead, gate, zn)
        # 注:曾试"病态标定守卫"(阈值漏过半fit缺陷→重标)治cable翻车,实测无效——病态是
        # fit/test漂移(fit缺陷强/test弱),fit侧看不见;守卫触发时反而重标低→pcb图级acc掉0.011。
        # 已撤,守卫类思路对这种"fit侧看不见test漂移"的病理普遍无效(seg_head/component_graph
        # 门控今天也撞了同一堵墙)——真正有效的是"永远融合"这种不依赖fit侧判断的确定性设计。

    #: 反馈硬约束最多允许把 fit 正常图的误报率推到这个绝对上限。
    #: 取10%的依据:生产实测误报率12.2%,若允许反馈把fit侧推到与之同量级,等于
    #: 把"救一张漏检"的代价放大到整条产线。10%是"还能救回大多数单点漏检、又不至于
    #: 让误报翻倍"的位置;超过它就宁可不救,并把不可满足如实告诉操作员。
    FB_FP_CAP = 0.10

    def _apply_feedback_constraint(self, ead, gate, zn):
        """**操作员反馈的硬约束**:人亲口标过的样本,必须判对。

        为什么要有:图级判决走的是 self._dino_thr,而它由 _calibrate(zn, zd) 在
        100张正常+30张缺陷上按平衡准确率选点。操作员反馈一张漏检,只是往那130票里
        加了1票——实测(scripts/verify_feedback.py, cable)阈值从 1.69098 变成
        1.69098,**小数点后五位都没动**,那张图反馈后依然漏检。分割头权重确实重训了
        (指纹 4f1d5f3d150d → 4ffabf2aee95),但操作员看到的"这张是不是缺陷"没变。
        赛题要求的是"操作员反馈→动态调整",不是"调了但judgement不变"。

        做法:把反馈样本从"投票"升格为"约束"。
          - 报过漏检的图 → 阈值必须 ≤ 它们的最低分(让它们全部过线)
          - 报过误检的图 → 阈值必须 > 它们的最高分(让它们全部不过线)
        两条约束都用 fit 正常图的误报率上限 FB_FP_CAP 兜底:**宁可救不回,也不让
        单张反馈把整条产线的误报推上去**。约束不可满足(比如两类反馈样本的分数区间
        交叉、或救它要付超过10%误报)时,退回平衡阈值,并把不可满足的样本记在
        self._fb_unsat 里——界面据此如实告诉操作员"这张图与正常样本在当前特征下
        不可分",而不是假装修好了。"""
        import numpy as np
        self._fb_unsat = None
        if not self._fb_defects and not self._fb_normals:
            return

        def _z(im):
            e = ead.score(im); d = gate.score(im)
            k = _assign_mode(gate.last_cls.detach().cpu(), self._modes)
            return float(_mode_z(e, d, self._modes[k]))

        zd_fb = [_z(x) for x in self._fb_defects]
        zn_fb = [_z(x) for x in self._fb_normals]
        zn = np.asarray(zn, dtype=float)
        base = float(self._dino_thr)

        lo = float(np.quantile(zn, 1.0 - self.FB_FP_CAP))   # 阈值不得低于此,否则误报超上限
        want_dn = min(zd_fb) - 1e-6 if zd_fb else None       # 漏检反馈要求阈值降到这
        want_up = max(zn_fb) + 1e-6 if zn_fb else None       # 误检反馈要求阈值升到这

        thr, unsat = base, []
        if want_dn is not None:
            if want_dn < lo:
                # **救不回就一动不动**。第一版在这里仍把阈值压到lo("尽力而为"),
                # 实测判负(scripts/verify_feedback.py, cable):fit口径只从7%涨到10%,
                # 留出正常图误报却 6.9%→24.1%(+17.2pp),召回+0,那张图照样漏——
                # fit正常图参与过标定、分布偏紧,fit口径的10%严重低估真实误报。
                # 付出17个点误报、换回0张召回,这钱不能花。
                unsat.append(f"{len(zd_fb)}张漏检反馈中最低分{min(zd_fb):.4g}低于误报上限"
                             f"对应的阈值{lo:.4g},救它要付>{self.FB_FP_CAP:.0%}误报,"
                             f"阈值维持不动")
            else:
                thr = min(thr, want_dn)
        if want_up is not None:
            if want_dn is not None and want_up >= want_dn:
                unsat.append(f"误检反馈最高分{max(zn_fb):.4g} ≥ 漏检反馈最低分"
                             f"{min(zd_fb):.4g},两类反馈样本在当前特征下不可分")
            else:
                thr = max(thr, want_up)
        self._dino_thr = thr
        self._fb_unsat = unsat or None
        print(f"!! 反馈硬约束: 阈值 {base:.5g} → {thr:.5g} "
              f"(漏检反馈{len(zd_fb)}张/误检反馈{len(zn_fb)}张, "
              f"fit正常图误报 {float((zn > base).mean()):.1%} → "
              f"{float((zn > thr).mean()):.1%})"
              + (f" ⚠️{unsat[0]}" if unsat else ""), flush=True)

    def _seg_stat(self, img):
        """把分割图压成一个图级标量:取前0.1%像素的均值(比单点max稳,比全图均值敏感)。
        2500²的256²图上0.1%≈65像素,正好是一处小缺陷的量级。"""
        import numpy as np
        a = self.segment(img).ravel()
        k = max(4, int(a.size * 0.001))
        return float(np.partition(a, -k)[-k:].mean())

    def _calibrate_seg_gate(self, normals, defects):
        """**用分割图当图级判据**(seg co-detector)。

        为什么加:2500²混合流实测,全局EAD图级分**完全失去区分力**——缺陷分中位5.09
        反而低于正常分中位5.37,两类重叠100%,任何阈值最好也只能到acc=0.699(=全判正常)。
        但**同一批数据上分割是好的**(框命中0.359/IoU 0.368)。
        原因很直接:图级分是全图统计出来的,而缺陷只占2500²面板的~0.1%,信号被稀释没了;
        分割头是逐像素的,还抓得到。既然像素级有信号而图级没有,就**从分割图里取图级判据**。

        代价:normal图不能再走"判正常立即返回"的早退路径(要先算分割图)。2500²上
        实测分割本来就在128ms的主路径里,预算200ms,付得起。"""
        import numpy as np
        ns = np.array([self._seg_stat(n) for n in normals[:40]])
        ds = np.array([self._seg_stat(d) for d in defects])
        mu, sd = ns.mean(), ns.std() + 1e-9
        zn = [float((x - mu) / sd) for x in ns]
        zd = [float((x - mu) / sd) for x in ds]
        thr = FewShotAdapter._calibrate(zn, zd)
        # SEG_GATE_STRICT:把阈值锚在"fit正常图的最大值"上(fit上零误报),而不是
        # 平衡准确率点。实测平衡准确率标定让误报从4.3%涨到12.9%、图级acc掉0.042;
        # 保守阈值应能保住大部分召回增益(92%→98%)而少付误报代价。
        import os as _os
        if _os.environ.get("SEG_GATE_STRICT") == "1":
            thr = max(thr, max(zn) + 1e-6)
        self._seg_gate = (mu, sd, thr)
        import os as _os
        if _os.environ.get("CALIB_DEBUG"):
            print(f"[seg-gate] 正常 中位={np.median(ns):.4g} max={ns.max():.4g} | "
                  f"缺陷 中位={np.median(ds):.4g} min={ds.min():.4g} | z阈值={thr:.4g} | "
                  f"重叠={sum(1 for x in ds if x <= ns.max())}/{len(ds)}", flush=True)

    def _calibrate_gcad_embed(self, normals, defects):
        """GCAD-EmbedAE标定(见aoi/gcad_embed.py docstring:已验证净正)。复用DINO门
        (self._dino)同一次前向顺手缓存的CLS token(last_cls,零增量前向),训一个瓶颈
        MLP自编码器判整图语义构图对不对。需要self._dino已标定成功才启用。"""
        import numpy as np
        self._embed_ae = None
        if len(normals) < 5:
            return

        def _cls_of(img):
            self._dino.score(img)                     # 顺手populate last_cls;fit不计时,可接受
            return self._dino.last_cls.cpu()

        cls_n = [_cls_of(n) for n in normals]
        cls_d = [_cls_of(d) for d in defects] if defects else []
        ae = fit_embed_ae(cls_n, device=self._bb_loc.device)
        mu, sd = _gcad_calibrate_zscore(ae, cls_n)

        def z(v):
            return (ae.score(v[None].to(self._bb_loc.device)) - mu) / sd

        zn = [z(c) for c in cls_n]
        if cls_d:
            zd = [z(c) for c in cls_d]
            thr = FewShotAdapter._calibrate(zn, zd)
        else:
            thr = float(np.mean(zn) + 3 * np.std(zn))  # 没有缺陷样本时退化为3-sigma
        self._embed_ae = ae
        self._embed_stats = (mu, sd)
        self._embed_thr = thr

    def _embed_score(self, img):
        """GCAD-EmbedAE的z-score,复用self._dino.last_cls(调用方须保证在predict()/
        self._dino.score(img)之后调用,否则last_cls是上一张图的、会读错)。"""
        if self._embed_ae is None or self._dino is None or self._dino.last_cls is None:
            return -1e9
        mu, sd = self._embed_stats
        raw = self._embed_ae.score(self._dino.last_cls[None].to(self._bb_loc.device))
        return (raw - mu) / sd

    def _select_feat_mode(self, defects, defect_masks, normals):
        """留出集(每4取1)对比 单特征 vs 模板差分特征(工业AOI金模板;pcb类刚性件+21%,
        非刚性中性),赢者定 extractor。差分推理多~30ms(ECC),仅在有效时启用。"""
        import numpy as np
        from .tmpl_ref import RefBank
        self._ref_bank = RefBank(normals)
        if len(defects) < 8:
            return                                          # 样本太少不选,保持单特征
        hold = list(range(0, len(defects), 4))
        tr = [i for i in range(len(defects)) if i not in set(hold)]

        def _try(extractor):
            # joint_ensemble要和生产头保持一致——否则是用"独立训"的规格选特征模式,
            # 但部署时用的是"联合训",选择依据和实际状态不匹配。
            h = SupervisedSegHead(device=self.seg_head.device, steps=150, extractor=extractor,
                                  joint_ensemble=getattr(self.seg_head, "joint_ensemble", False))
            ok = h.fit(self._eff(), [defects[i] for i in tr], [defect_masks[i] for i in tr],
                       normals[:15])
            if not ok or h.thr is None:
                return -1.0
            ious = []
            for i in hold:
                amap = h.map(self._eff(), defects[i], self.seg_eval_hw)
                gt = _mask_np(defect_masks[i], self.seg_eval_hw)
                pred = amap >= h.thr
                TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(ious))

        # 候选:WRN单特征(基线,最快)vs 模板差分(工业金模板,pcb类+21%)。
        # DINO组件语义特征曾试(逻辑定位):64²探针假象,原生口径WRN分辨率优势胜出,已撤(见run_logic_native.py)。
        iou_single = _try(self._wrn_feats)
        iou_diff = _try(self._wrn_feats_diff)
        if iou_diff > iou_single + 0.01:                    # 差分要赢出margin才启用(它更贵)
            self.seg_head.extractor = self._wrn_feats_diff
            self.feat_mode = "tmpl_diff"
        else:
            self.seg_head.extractor = self._wrn_feats
            self.feat_mode = "single"

    def _select_seg_head_style(self, defects, defect_masks, normals, k=4):
        """【已停用,fit_fewshot不再调用——留档记录负结果】k折CV对比新式seg_head(4头bagging+
        soft loss+OOF-IoU阈值)vs旧式(双头+pooled-F1,ae5fbbb存档于_seg_head_old_ae5fbbb.py)。
        动机:真实3类AD2 A/B(run_seg_head_ab.py)暴露均值净平(0.598≈0.598)掩盖逐类反号大方差
        (sheet_metal新+0.111 / walnuts旧+0.020 / fruit_jelly旧+0.092),按类选优理论上限≈0.635。
        结果:单次留出split版3/3选错;升级4折CV后仍3/3选新(fruit_jelly真实差距0.092也没抓住)
        ——不是split噪声,是fit/test分布漂移:新式在fit分布held-out上真不差,回归只在test分布
        出现,fit侧任何CV原理上都看不见。门控挣不到位置(永远选默认+浪费8次fit),已从
        fit_fewshot撤下;新式保持唯一默认(均值持平,且赢的sheet_metal细小缺陷/工业表面
        最接近赛题手机件场景,新式又直接优化赛题主指标逐图IoU)。"""
        import numpy as np
        from ._seg_head_old_ae5fbbb import SupervisedSegHead as _OldSegHead
        if len(defects) < 8:
            return
        extractor = self.seg_head.extractor
        rams_extractor = self.seg_head.rams_extractor
        n = len(defects)
        kk = min(k, max(2, n // 4))
        order = list(range(n))
        import random as _r
        _r.Random(0).shuffle(order)
        folds = [order[i::kk] for i in range(kk)]

        def _cv_iou(cls):
            all_ious = []
            for fi in range(kk):
                hold = folds[fi]
                tr = [i for i in range(n) if i not in set(hold)]
                h = cls(device=self.seg_head.device, extractor=extractor, rams_extractor=rams_extractor)
                ok = h.fit(self._eff(), [defects[i] for i in tr], [defect_masks[i] for i in tr], normals[:15])
                thr = getattr(h, "thr", None)
                if not ok or thr is None:
                    continue
                for i in hold:
                    amap = h.map(self._eff(), defects[i], self.seg_eval_hw)
                    gt = _mask_np(defect_masks[i], self.seg_eval_hw)
                    pred = amap >= thr
                    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                    all_ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(all_ious)) if all_ious else -1.0

        iou_new = _cv_iou(SupervisedSegHead)
        iou_old = _cv_iou(_OldSegHead)
        if iou_old > iou_new + 0.01:                       # 旧式要赢出margin才切换(新式默认更值得信任)
            self.seg_head = _OldSegHead(device=self.seg_head.device, extractor=extractor,
                                         rams_extractor=rams_extractor)
            self.seg_head_style = "old_ensemble_pooledF1"
        else:
            self.seg_head_style = "new_bagging_oofIoU"

    @torch.no_grad()
    def _wrn_feats_diff(self, img):
        """模板差分特征:concat[feat(test), feat(test)-feat(ECC对齐最近邻正常参考)]。"""
        f = self._wrn_feats(img)
        ref = self._ref_bank.aligned_ref(img if img.dim() == 3 else img[0])
        _on = getattr(self, "_wrn_cache_on", False)
        self._wrn_cache_on = False        # 双保险:参考图是另一张图,绝不能碰测试图的缓存
        try:
            fr = self._wrn_feats(ref)
        finally:
            self._wrn_cache_on = _on
        return torch.cat([f, f - fr], dim=0)

    def _calibrate_boxes(self, defects, defect_masks):
        """在fit缺陷上搜碎框合并距离d(最大化GT框召回@0.5)。实测电子件+0.02~0.04。"""
        import numpy as np
        from .seg_head import merge_boxes
        try:
            import cv2
        except Exception:
            self.box_merge_d = 0; return
        if self.seg_head.thr is None:
            self.box_merge_d = 0; return

        def gtb(mk):
            m = cv2.resize(mk.astype(np.uint8), self.seg_eval_hw[::-1], interpolation=cv2.INTER_NEAREST)
            n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            return [(x, y, x + w, y + h) for x, y, w, h, a in (st[i] for i in range(1, n)) if a >= 4]

        def biou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1)

        preds, gts = [], []
        for img, mk in zip(defects, defect_masks):
            amap = self.segment(img)
            preds.append(map_to_boxes((amap >= self.seg_head.thr).astype(np.float32), 0.5,
                                      min_area_frac=0.0002, close=0))
            gts.append(gtb(mk))
        best_d, best_h = 0, -1
        for d in [0, 4, 8, 16]:
            tot = hit = 0
            for pb, gb in zip(preds, gts):
                mb = merge_boxes(pb, d)
                for g in gb:
                    tot += 1
                    if any(biou(p[:4], g) >= 0.5 for p in mb):
                        hit += 1
            h = hit / max(tot, 1)
            if h > best_h:
                best_h, best_d = h, d
        self.box_merge_d = best_d

    def _native_crops(self, imgs, masks, pad=1.0, min_c=256, max_c=1024):
        """对每张缺陷图:按掩膜连通域在原生分辨率裁块(带上下文),供分割头训练。
        小图(长边<700)原生≈全局视角,跳过。返回 (crop_imgs, crop_masks)。"""
        import numpy as np
        try:
            import cv2
        except Exception:
            return [], []
        ci, cm = [], []
        for img, mk in zip(imgs, masks):
            H, W = img.shape[-2:]
            if max(H, W) < 700:
                continue
            mh, mw = mk.shape
            n, _, stats, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
            for i in range(1, min(n, 4)):
                x, y, w, h, a = stats[i]
                if a < 2:
                    continue
                # 掩膜坐标 → 原图坐标
                cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
                side = int(min(max(max(w * W / mw, h * H / mh) * (1 + 2 * pad), min_c), max_c))
                x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
                y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
                x1, y1 = min(W, x0 + side), min(H, y0 + side)
                crop = img[:, y0:y1, x0:x1]
                # 掩膜对应子区域(掩膜空间)
                mx0, my0 = int(x0 * mw / W), int(y0 * mh / H)
                mx1, my1 = max(mx0 + 1, int(x1 * mw / W)), max(my0 + 1, int(y1 * mh / H))
                sub = mk[my0:my1, mx0:mx1]
                if sub.sum() == 0:
                    continue
                ci.append(crop); cm.append(sub.astype(np.uint8))
        return ci, cm

    def _eff(self):
        """底层 EfficientADDetector(residual_map_large/anomaly_map_large 在它上面)。"""
        return self.branches[0].det.det

    def _calibrate_pixel(self, normals):
        """像素二值阈值:优先用监督头在fit缺陷掩膜上标的F1最优阈值(实测IoU +58%);
        无监督头(无掩膜)则回退正常p99.5分位。"""
        import numpy as np
        if self.seg_head.thr is not None:
            self.pix_thr = self.seg_head.thr
            return
        vals = [self.segment(n).ravel() for n in normals[:20]]
        self.pix_thr = float(np.quantile(np.concatenate(vals), 0.995))

    def _segment_once(self, img):
        """单次前向的像素级异常图(无TTA)。"""
        eff = self._eff()
        sup = self.seg_head.map(eff, img, self.seg_eval_hw)
        amap = sup if sup is not None else eff.anomaly_map_large(img, out_hw=self.seg_eval_hw)
        if self.boundary_refiner is not None:
            amap = self.boundary_refiner.refine(self, img, amap)  # DCP-SFR边界残差(fit留出验证净正才启用)
        return amap

    def segment(self, img):
        """像素级异常图(原始尺度,不逐图标准化→阈值语义清晰)。
        有监督头(迁移带掩膜)→用它的 logit(BCE训,>0≈缺陷,实测均值0.890,救弱项);
        无掩膜→回退无监督 EAD 异常图。
        TTA(测试时增强,self.use_tta):水平翻转取logit均值。和今天判负的几条(UniVAD v2/
        DCP-SFR/crop_cascade)风险性质不同——那几条是"学一个小修正模块,靠fit留出少量
        数据判断该不该信",连续撞了三次"fit侧判不准test侧"的墙;TTA是确定性多视角平均,
        不需要任何学习/门控决策,原理上不会因为fit数据不够而误判。仍需真实数据验证净
        增益(见run_tta_ab.py),默认关。"""
        amap = self._segment_once(img)
        if getattr(self, "use_tta", False):
            import numpy as np
            native = img if img.dim() == 3 else img[0]
            flipped = torch.flip(native, dims=[-1])
            amap2 = self._segment_once(flipped)
            amap2 = np.flip(amap2, axis=-1).copy()            # 翻回原方向对齐
            amap = (amap + amap2) / 2.0
        return amap

    def _pf(self, stage, t0):
        """逐段计时(LOCATE_PROFILE=1 才启用)。GPU是异步的,必须synchronize才能测准,
        所以只在剖析模式下开——正常推理路径零开销。"""
        import time as _t
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.prof.setdefault(stage, []).append((_t.perf_counter() - t0) * 1000)
        return _t.perf_counter()

    def locate(self, img):
        """完整定位输出:图级分(EAD)+ 判决 + 类型 + 像素图 + 检测框。
        延时热路径:EAD/DINO判正常且不处于救援灰区→立即返回,不算WRN分割/SAM/crop级联
        (它们只有判缺陷才有意义,正常图是生产大多数,省的是主要延时)。

        外壳只负责给WRN特征缓存划定作用域:入口开、finally关(异常路径也保证关闭)。
        fit期不走这里,所以缓存绝不会跨图污染训练——见 _wrn_feats 的说明。"""
        self._wrn_cache_on = True
        self._wrn_cache = None
        try:
            return self._locate_inner(img)
        finally:
            self._wrn_cache_on = False
            self._wrn_cache = None

    def _locate_inner(self, img):
        import numpy as np
        import os as _os, time as _time
        _prof = _os.environ.get("LOCATE_PROFILE") == "1"
        if _prof:
            if not hasattr(self, "prof"):
                self.prof = {}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t = _time.perf_counter()
        if torch.cuda.is_available() and img.device.type == "cpu":
            img = img.to(self._bb_loc.device)   # 单次H2D上传:下游EAD/DINO/WRN/aux的.to()全变no-op
        if _prof:
            _t = self._pf("1_上传", _t)
        res = self.predict(img)
        if _prof:
            _t = self._pf("2_图级判决(EAD+DINO)", _t)
        # 受控补检v6:EAD判正常但处于自适应灰区(score≥g×阈值,g由fit B半联合验证选定)
        # 且seg信号超线→翻正。g与线在fit上按"联合翻正条件零误翻"标定,守不住则整体禁用。
        g = getattr(self, "rescue_gray", None)
        rescue_zone = (not res["is_defect"] and g is not None
                       and getattr(self, "rescue_seg_thr", None) is not None
                       and self.threshold is not None
                       and res["score"] >= g * self.threshold)
        # GCAD-EmbedAE OR门:base(EAD+DINO)+灰区补检都判"正常"时才查,独立阈值,只能
        # 新增覆盖、不改变base原有判定边界(self._dino.last_cls是predict()里self._dino.score()
        # 同一次前向顺手缓存的,这里直接读,零增量前向)。已验证净正,见aoi/gcad_embed.py。
        gcad_trigger = (not res["is_defect"] and not rescue_zone
                        and self._embed_ae is not None
                        and self._embed_score(img) >= self._embed_thr)
        if not res["is_defect"] and not rescue_zone and not gcad_trigger:
            res["anomaly_map"] = None; res["mask"] = None; res["boxes"] = []
            if _prof:
                self.prof.setdefault("0_早退", []).append(1)
            return res
        amap = self.segment(img)
        if _prof:
            _t = self._pf("3_分割头", _t)
        thr = self.pix_thr if self.pix_thr is not None else float(amap.mean() + 3 * amap.std())
        res["anomaly_map"] = amap
        if rescue_zone and float(amap.max()) >= self.rescue_seg_thr:
            res["is_defect"] = True
            res["rescued"] = "seg"
            raws = res["_raws"] if res["_raws"] is not None else ([res["score"]] + self._aux_raws(img))
            res["defect_type"] = self._ztype(raws)
        if gcad_trigger and not res["is_defect"]:
            res["is_defect"] = True
            res["rescued"] = "gcad"
            raws = res["_raws"] if res["_raws"] is not None else ([res["score"]] + self._aux_raws(img))
            res["defect_type"] = self._ztype(raws)
        if res["is_defect"]:
            mask = (amap >= thr).astype(np.uint8)
            if self.roi_zoom:
                mask = self._zoom_refine(img if img.dim() == 3 else img[0], mask, thr)  # 原生裁块重分割
            if self.sam is not None:
                mask = self.sam.refine(img if img.dim() == 3 else img[0], mask, amap=amap)  # SAM受控精化(逐区域OOF)
                if _prof:
                    _t = self._pf("4_SAM精化", _t)
            if self.crop_cascade is not None:
                mask = self.crop_cascade.refine(self, img if img.dim() == 3 else img[0], mask, mask.shape)  # 独立crop-head补微小缺陷
            if self.comp_graph is not None:
                mask = self.comp_graph.refine(self, img, mask)          # 组件图补逻辑缺陷(缺件/错位)
            res["mask"] = mask
            # 掩膜已阈值化+SAM精化,面积门槛放宽到~13px(默认52px会滤掉pcb类5×5微小缺陷)
            from .seg_head import merge_boxes
            boxes = map_to_boxes(mask.astype(np.float32), 0.5, min_area_frac=0.0002, close=0)
            res["boxes"] = merge_boxes(boxes, getattr(self, "box_merge_d", 0))
            if not res["boxes"]:
                # **判了缺陷就必须给出位置**。掩膜阈值化后可能一个像素都不超线(尤其
                # 灰区救援/边界样本),此时输出 is_defect=1 却给空框——赛题明确要求
                # 画出缺陷框,声明缺陷却答不出位置是不完整的答案。
                # 退回:取异常图响应最高的那一小撮像素的外接框(至少给出最可疑区域)。
                # 取最高响应处的**最大连通域**外接框,不能直接用top-k的外接框——
                # top-k像素会被噪声散布到全图,外接框撑到接近整图,和不给框一样没用
                # (实测:256²图上top-32的外接框是(4,56,241,241))。
                import cv2 as _cv2
                hi = (amap >= np.percentile(amap, 99.9)).astype(np.uint8)
                n_, _, st_, _ = _cv2.connectedComponentsWithStats(hi, connectivity=8)
                if n_ > 1:
                    j = 1 + int(np.argmax(st_[1:, _cv2.CC_STAT_AREA]))
                    x, y, w, h = st_[j, :4]
                    res["boxes"] = [(int(x), int(y), int(x + w), int(y + h),
                                     float(amap[y:y + h, x:x + w].max()))]
                else:                                      # 极端兜底:最高点周围一个小窗
                    yy, xx = np.unravel_index(int(np.argmax(amap)), amap.shape)
                    r = max(2, min(amap.shape) // 32)
                    res["boxes"] = [(int(max(0, xx - r)), int(max(0, yy - r)),
                                     int(min(amap.shape[1], xx + r)), int(min(amap.shape[0], yy + r)),
                                     float(amap.max()))]
            if _prof:
                _t = self._pf("5_出框", _t)
            # VLM监督的类型头(fit期蒸馏,推理零API)。它要掩膜才能算位置匹配特征,
            # 所以只能放在掩膜产出之后;不可用时res["defect_type"]保持_ztype的启发式结果。
            th = getattr(self, "type_head", None)
            if th is not None and th.ready:
                t = th.predict(self, img, mask, res.get("_raws"))
                if t is not None:
                    res["defect_type"] = t
                if _prof:
                    _t = self._pf("6_类型头", _t)
        else:
            res["mask"] = None
            res["boxes"] = []
        return res

    def _zoom_refine(self, img, mask, thr, pad=1.0, min_c=256, max_c=1024, max_regions=6):
        """2500²大图定位精化:粗掩膜连通域→原生分辨率裁块→分割头重打分→贴回。
        消除'全图下采样512把微小缺陷抹掉'的结构损失;小图(长边<700)直接返回。"""
        import numpy as np
        try:
            import cv2
        except Exception:
            return mask
        H, W = img.shape[-2:]
        if max(H, W) < 700 or self.seg_head.head is None:
            return mask
        mh, mw = mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        out = mask.copy()
        for i in range(1, min(n, max_regions + 1)):
            x, y, w, h, a = stats[i]
            if a < 2:
                continue
            cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
            side = int(min(max(max(w * W / mw, h * H / mh) * (1 + 2 * pad), min_c), max_c))
            x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
            y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
            x1, y1 = min(W, x0 + side), min(H, y0 + side)
            sub_amap = self.seg_head.map(self._eff(), img[:, y0:y1, x0:x1], (128, 128))
            sub_mask = (sub_amap >= thr).astype(np.uint8)
            # 贴回掩膜空间对应区域
            mx0, my0 = int(x0 * mw / W), int(y0 * mh / H)
            mx1, my1 = max(mx0 + 1, int(x1 * mw / W)), max(my0 + 1, int(y1 * mh / H))
            sub_rs = cv2.resize(sub_mask, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
            out[my0:my1, mx0:mx1] = sub_rs
        return out

    def _ztype(self, raws):
        """各分支 z 分,最高者定缺陷类型(z 归一后可比)。"""
        zs = [znorm(r, m, s) for r, (m, s) in zip(raws, self.stats)]
        return self.branches[max(range(len(zs)), key=lambda i: zs[i])].defect_type

    def _aux_raws(self, img):
        """3路轻辅助分支(色彩/尺寸/结构)分,仅判缺陷才需要(定类型)——共享一次下采样。
        显式回CPU:分支的统计bank是fit期在CPU建的(色彩bank等),GPU张量会cdist设备不匹配;
        320²小图回传开销可忽略,单次上传的收益留给EAD/DINO/WRN大头。"""
        aux_x = _down(img, self.aux_size).cpu()
        return [b.score(img, pre=aux_x) for b in self.branches[1:]]

    def predict(self, img):
        score = self.branches[0].score(img)               # 检测分 = EAD 核心(唯一检测层,无条件算)
        is_def = bool(self.threshold is not None and score >= self.threshold)
        if getattr(self, "_dino", None) is not None:      # DINOv2 图级co-detector:平等融合门
            dsc = self._dino.score(img)
            ms = getattr(self, "_modes", None)
            if ms and len(ms) > 1:                        # per-mode:先归模态,再用该模态统计量
                m = ms[_assign_mode(self._dino.last_cls.detach().cpu(), ms)]
                fused = float(_mode_z(score, dsc, m))
            else:
                fused = self._dino_fuse(score, dsc)
            is_def = bool(fused >= self._dino_thr)
        sg = getattr(self, "_seg_gate", None)
        if sg is not None:                                # seg co-detector:像素级信号兜住图级
            mu, sd, sthr = sg
            if float((self._seg_stat(img) - mu) / sd) >= sthr:
                is_def = True
        if is_def:
            raws = [score] + self._aux_raws(img)          # 类型归属(3路辅助分支)只在判缺陷时算,省正常图开销
            return {"score": score, "is_defect": True, "defect_type": self._ztype(raws), "_raws": raws}
        return {"score": score, "is_defect": False, "defect_type": "normal", "_raws": None}

    def frame_score(self, img):
        """逐帧检测分(视频路径用,与 predict() 图级门同口径):EAD核心分,DINO门启用时
        返回 max(z_EAD,z_DINO) 融合分。配 decision_threshold() 时序平滑判决,让视频也吃到
        受控DINO co-detector(此前视频直接用EAD分绕过了门)。"""
        ead = self.branches[0].score(img)
        if getattr(self, "_dino", None) is not None:
            return self._dino_fuse(ead, self._dino.score(img))
        return ead

    def decision_threshold(self):
        """当前图级判决阈值:DINO门启用→融合阈值,否则EAD阈值。与 frame_score() 配对。"""
        if getattr(self, "_dino", None) is not None:
            return self._dino_thr
        return self.threshold

    def explain(self, img):
        """回溯检测逻辑(赛题"用户反馈驱动的优化"明确要求"系统应能回溯检测逻辑"):
        把这张图走过的完整判定链路摊开——每个分支的原始分、标准化后的z分、生效的
        阈值、是哪一步做出的判定、缺陷类型怎么定的、掩膜经过哪些精化模块。

        **不在热路径上**:locate()一行未改,explain()是操作员事后复盘时才调用的冷
        路径(单独重算一遍),不占200ms预算。返回纯python字典,可直接json化给前端/
        报告用。"""
        import numpy as np
        if torch.cuda.is_available() and img.device.type == "cpu":
            img = img.to(self._bb_loc.device)
        trace = {}

        # ① 图级检测:EAD核心分(唯一无条件算的检测层)
        ead_raw = float(self.branches[0].score(img))
        trace["ead"] = {"raw": ead_raw, "threshold": self.threshold,
                        "超阈值": bool(self.threshold is not None and ead_raw >= self.threshold)}

        # ② DINO门(默认永远融合):max(z_EAD, z_DINO)联合判定
        if getattr(self, "_dino", None) is not None:
            dino_raw = float(self._dino.score(img))
            emu, esd, dmu, dsd = self._dino_stats
            z_ead, z_dino = (ead_raw - emu) / esd, (dino_raw - dmu) / dsd
            fused = float(self._dino_fuse(ead_raw, dino_raw))
            trace["dino"] = {"raw": dino_raw, "z_ead": float(z_ead), "z_dino": float(z_dino),
                             "fused(取大)": fused, "threshold": float(self._dino_thr),
                             "谁主导": "DINO" if z_dino > z_ead else "EAD"}
            is_def = fused >= self._dino_thr
            trace["图级判定依据"] = "EAD+DINO融合门"
        else:
            is_def = bool(self.threshold is not None and ead_raw >= self.threshold)
            trace["dino"] = None
            trace["图级判定依据"] = "纯EAD(DINO门未启用)"
        trace["图级判定"] = bool(is_def)

        # ③ 灰区补检 / GCAD OR门(两条可能翻正的救援路径,如果启用)
        g = getattr(self, "rescue_gray", None)
        if not is_def and g is not None and getattr(self, "rescue_seg_thr", None) is not None \
                and self.threshold is not None:
            in_zone = ead_raw >= g * self.threshold
            trace["灰区补检"] = {"处于灰区": bool(in_zone), "灰区系数g": float(g),
                                 "灰区下界": float(g * self.threshold)}
        if not is_def and self._embed_ae is not None:
            z_emb = float(self._embed_score(img))
            trace["GCAD语义OR门"] = {"z": z_emb, "threshold": float(self._embed_thr),
                                      "独立触发": bool(z_emb >= self._embed_thr)}

        # ④ 缺陷类型归属(3路辅助分支z分竞争,最高者定类型)
        if is_def:
            raws = [ead_raw] + self._aux_raws(img)
            zs = [float(znorm(r, m, s)) for r, (m, s) in zip(raws, self.stats)]
            trace["类型归属"] = {
                "各分支z分": {b.defect_type: z for b, z in zip(self.branches, zs)},
                "判定类型": self.branches[int(np.argmax(zs))].defect_type,
            }

        # ⑤ 定位链路:掩膜依次经过哪些模块(只有判缺陷才走)
        o = self.locate(img)
        trace["定位"] = {
            "像素阈值": float(self.pix_thr) if self.pix_thr is not None else None,
            "分割头": "监督头(有掩膜训练)" if self.seg_head.head is not None else "回退EAD无监督异常图",
            "特征模式": getattr(self, "feat_mode", "single"),
            "精化模块": [n for n, on in [("roi_zoom", self.roi_zoom),
                                          ("SAM边界精化", self.sam is not None),
                                          ("crop级联", self.crop_cascade is not None),
                                          ("组件图", self.comp_graph is not None)] if on],
            "输出框数": len(o.get("boxes") or []),
            "掩膜前景像素": int(o["mask"].sum()) if o.get("mask") is not None else 0,
        }
        trace["最终判定"] = {"is_defect": bool(o["is_defect"]),
                             "defect_type": o["defect_type"],
                             "救援路径": o.get("rescued")}
        # 延时自适应把哪些模块裁掉了(评委真机上会影响判定链,必须可回溯)
        trace["延时自适应裁剪"] = list(getattr(self, "lat_trimmed", []) or [])
        return trace
