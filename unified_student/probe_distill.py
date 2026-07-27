"""低成本探针(统一骨干提案的第一步,建完整YOLO-P2四头学生之前的止损点):
验证"WRN特征本身能不能替代EAD+DINO这两个独立模型的图级判决"。

不碰真实缺陷标签——纯粹是判决蒸馏:用已经产出的EAD+DINO融合判决(det.predict()的
真实输出,生产代码本来就在算)当"教师信号",在已经算出来的WRN特征(seg_head本来
就要用的那份)上加一个极小MLP头去学着预测这个融合分数。只在fit数据上训+标定阈值,
在真正held-out的test图上看蒸馏头自己的判决和EAD+DINO真实判决的**一致率**——这是
唯一有意义的检验,fit集上做到的一致率没有参考价值(EAD/DINO阈值本来就是拿fit集
标定出来的,一致率虚高)。

成功判据(动手前定好,不是看完结果再找理由):
  多类目一致率普遍>0.90 → WRN特征扛得住EAD+DINO在做的事,统一骨干的核心假设
    成立,值得投入建完整YOLO-P2四头学生架构。
  一致率偏低或类目间大幅波动 → WRN特征(至少现在这个512²/128²分辨率)信息量不够,
    统一骨干这条路要么止损,要么至少要先解决"用什么特征表示图级判决"这个问题。

【已验证,判负】两版聚合方式(mean pooling / top-k pooling,均见下方代码)在
breakfast_box/pushpins/cable/pcb四类上跑完:mean pooling一致率median=0.724
mean=0.735 min=0.691;top-k一致率反而更差(median=0.674),且按类目分化明显
(cable/pcb这类局部集中缺陷top-k更好,breakfast_box/pushpins这类逻辑异常top-k
反而更差,pushpins相关系数从0.925暴跌到0.163)。两版都远低于0.90门槛。
最damning的证据是pcb:mean pooling下相关系数只有0.068(WRN特征几乎不携带
EAD+DINO判决相关信息),cable上学生漏判3/15(恰是DINO独立信号救回来的那部分)。
**结论:复用现有冻结WRN特征做统一图级判决这条捷径不可行,不代表专门蒸馏新骨干
(不复用现有特征)也一定失败,但排除了"现成特征+小头"这个最低成本路线。**

用法:PYTHONPATH=. python unified_student/probe_distill.py(封存实验,mean/topk
两版代码都保留在下方,不再新增第三种聚合方式)
"""
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class TinyDistillHead(nn.Module):
    """逐位置打分(1×1 conv,C→32→1),不是先把整图特征平均池化成一个向量再回归。
    第一版用全局平均池化的结果是相关系数在pcb上几乎为0(0.068)——mean pooling会把
    一小块异常patch的信号被一整张图的正常背景平均稀释掉,DINO/EAD自己都不是这么
    聚合的(DinoGate.score()用的是"top-1%最差patch残差均值",不是全图均值)。
    这里改成逐位置出分再top-k聚合,和EAD/DINO自己的聚合方式保持一致,才是公平对比。"""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, 32, 1), nn.ReLU(True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        return self.net(x)                       # (B,1,h,w) 逐位置分数图


def _topk_agg(score_map, topq=0.01):
    """(1,h,w)分数图→标量,top-1%最高位置均值(同DinoGate.score()的聚合口径)。"""
    flat = score_map.reshape(-1)
    k = max(1, int(flat.numel() * topq))
    return torch.topk(flat, k).values.mean()


@torch.no_grad()
def _wrn_feat(det, img):
    return det._wrn_feats(img)                  # (C,h,w),复用seg_head本来就要算的特征,不做池化


def _teacher_score(det, img):
    """EAD+DINO融合分数(生产predict()内部真实用的那套逻辑,det._dino为None时纯EAD)。"""
    score = det.branches[0].score(img)
    if det._dino is not None:
        return det._dino_fuse(score, det._dino.score(img))
    return score


def run_one(name, normals, fit_i, fit_m, test_defs):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det._dino is None:
        # 今天GPU连轴跑一天,_calibrate_latency硬线超时会把已标定好的DINO门砍掉
        # (见CLAUDE.md今天的记录)——这里探针要测的是"EAD+DINO融合判决",不是延时
        # 预算,强制重新标定,避免被同一个confound污染结果。
        det._calibrate_dino_gate(normals, fit_i)

    # 训练/标定池:fit的正常图+缺陷图,各自的教师分数当回归目标
    train_imgs = normals[:60] + fit_i
    with torch.no_grad():
        train_feats = torch.stack([_wrn_feat(det, im) for im in train_imgs]).to(DEV)  # (N,C,h,w)
        train_targets = torch.tensor([_teacher_score(det, im) for im in train_imgs],
                                     device=DEV, dtype=torch.float32)
    C = train_feats.shape[1]
    mu = train_feats.mean(dim=(0, 2, 3), keepdim=True)
    sd = train_feats.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    torch.manual_seed(0)
    head = TinyDistillHead(C).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    n = train_feats.shape[0]
    for _ in range(300):
        idx = torch.randint(0, n, (min(16, n),), generator=g)
        score_maps = head((train_feats[idx] - mu) / sd)                     # (b,1,h,w)
        pred = torch.stack([_topk_agg(sm) for sm in score_maps])
        loss = ((pred - train_targets[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    # 阈值只在fit池上标定(和生产纪律一致):蒸馏头自己的分数分布，找出能重现教师
    # is_defect决策的最优切点（用教师在fit池上的真实判决当参照，F1最优）
    with torch.no_grad():
        train_pred = torch.stack([_topk_agg(sm) for sm in head((train_feats - mu) / sd)]).cpu().numpy()
    teacher_thr = det.decision_threshold()
    teacher_is_def_fit = (train_targets.cpu().numpy() >= teacher_thr).astype(int)
    order = np.argsort(-train_pred)
    ls = teacher_is_def_fit[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    student_thr = float(train_pred[order][int(np.argmax(f1))])

    # 真正的检验:held-out test图上,学生判决 vs 教师真实判决 一致率
    test_imgs = [img for img, _ in test_defs]
    with torch.no_grad():
        test_feats = torch.stack([_wrn_feat(det, im) for im in test_imgs]).to(DEV)
        student_pred = torch.stack([_topk_agg(sm) for sm in head((test_feats - mu) / sd)]).cpu().numpy()
    teacher_scores = np.array([_teacher_score(det, im) for im in test_imgs])
    teacher_is_def = teacher_scores >= teacher_thr
    student_is_def = student_pred >= student_thr
    agree = float((teacher_is_def == student_is_def).mean())
    corr = float(np.corrcoef(teacher_scores, student_pred)[0, 1]) if len(test_imgs) > 1 else float("nan")
    print(f"{name:24s} 一致率={agree:.3f}  分数相关系数={corr:.3f}  "
          f"(教师判缺陷{int(teacher_is_def.sum())}/{len(test_imgs)},"
          f"学生判缺陷{int(student_is_def.sum())}/{len(test_imgs)})", flush=True)
    return agree, corr


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
        ("生产:cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("生产:pcb", lambda: prep_realiad("pcb")),
    ]
    agrees = []
    for name, prep in jobs:
        agree, corr = run_one(name, *prep())
        agrees.append(agree)
    a = np.array(agrees)
    print(f"\n=== 汇总(n={len(a)}) === 一致率 median={np.median(a):.3f} mean={np.mean(a):.3f} min={np.min(a):.3f}",
          flush=True)
    print("判据:median>0.90且min>0.80 → WRN特征扛得住EAD+DINO判决,值得投入建完整架构;"
          "否则止损,不建YOLO-P2四头学生。", flush=True)


if __name__ == "__main__":
    main()
