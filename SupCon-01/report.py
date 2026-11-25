"""Recompute paired diagnostics from UID evidence, then render Chinese reports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=("S0","S1")
SEEDS=("42","43","44")


def read(filename):
    return json.loads((ROOT/filename).read_text())


def table(headers,rows):
    return "\n".join("| "+" | ".join(map(str,row))+" |" for row in [headers,["---"]*len(headers),*rows])


def fmt(v):
    return f"{v:.4f}"


def vector(v):
    return " / ".join(map(str,v))


def pct(v):
    return f"{100*v:.1f}%"


def markdown(name,lines):
    (ROOT/name).write_text("\n\n".join(lines)+"\n",encoding="utf-8")


def verify_and_interval(results):
    predictions={}
    for s in SEEDS:
        for arm in ARMS:
            tag=f"{s}_{arm}"
            saved=np.load(ROOT/f"centroids_{tag}.npz")
            with (ROOT/f"retrieval_{tag}.csv").open() as f:
                neighbors=list(csv.DictReader(f))
            sims=saved["val"]@saved["train"].T
            assert not set(saved["val_uid"])&set(saved["train_uid"])
            nn=sims.argmax(1)
            assert [int(row["neighbor_uid"]) for row in neighbors]==saved["train_uid"][nn].tolist()
            assert np.mean(saved["val_label"]==saved["train_label"][nn])==results[s][arm]["embedding"]["nearest"]
            for split in ("train","val"):
                with (ROOT/f"uid_{tag}_{split}.csv").open() as f:
                    rows=list(csv.DictReader(f))
                y=np.array([int(r["label"]) for r in rows])
                p=np.array([np.argmax([float(r["p_neg"]),float(r["p_neu"]),float(r["p_pos"])]) for r in rows])
                assert np.array_equal(y,saved[split+"_label"])
                cm=np.bincount(3*y+p,minlength=9).reshape(3,3)
                assert cm.tolist()==results[s][arm][split]["cm"]
                if split=="val":
                    predictions[s,arm]=p
    # Resample the same UIDs across models and seeds; keep the observed class counts.
    rng=np.random.default_rng(1943)
    ix=np.concatenate([rng.choice(np.flatnonzero(y==c),(5000,int((y==c).sum())),replace=True)
                       for c in range(3)],axis=1)
    def metrics(p):
        cm=np.stack([(p[ix]==j)&(y[ix]==i) for i in range(3) for j in range(3)]).sum(-1).T.reshape(-1,3,3)
        tp=np.diagonal(cm,axis1=1,axis2=2)
        return np.stack([(2*tp/(cm.sum(1)+cm.sum(2))).mean(1),(tp/cm.sum(2)).mean(1)])
    delta=np.mean([metrics(predictions[s,"S1"])-metrics(predictions[s,"S0"]) for s in SEEDS],axis=0)
    return np.quantile(delta,[.025,.975],axis=1).T


def main():
    r,a,data,smoke=map(read,["results.json","sampling.json","preflight.json","smoke.json"])
    ci=verify_and_interval(r)
    rows=[(s,k,r[s][k]) for s in SEEDS for k in ARMS]
    audit=[]
    for s in SEEDS:
        b=np.load(ROOT/f"batches_{s}.npy")
        assert b.shape==(52,63)
        assert hashlib.sha256((ROOT/f"batches_{s}.npy").read_bytes()).hexdigest()==a[s]["sha256"]
        audit.append([s,a[s]["distinct_uids"],a[s]["distinct_trials"],vector(
            f"{lo}–{hi}" for lo,hi in a[s]["uid_uses_by_class"]),
            vector(f"{n:.2f}" for n in a[s]["mean_words_by_class"]),a[s]["sha256"][:16]])
    markdown("BATCH_SAMPLING_AUDIT.md",[
        "# 数据、成对采样与 smoke 审计",
        "两份缓存的 x 来自冻结 V3 epoch 46 的纯 EEG 路径。其生成代码只将 EEG、mask 和既有提示传入 encoder，文本情感标签另存于 label。运行前核对了原 checkpoint 文件哈希、冻结缓存的预检记录，以及 train/val 的逐 UID 规范句、标签和 trial 数。逐 trial 的 UID、被试及标签也与上轮 assignments 文件一致。两份缓存均只把 x 提供给新模型；句子只用于统计长度和核对来源。本轮未构建 Test 数据集。",
        table(["划分","trial × 特征维度","UID 类别数 N/U/P","trial 类别数 N/U/P","被试数","平均词数 N/U/P"],
              [[k,vector(data[k]["features"]),vector(data[k]["class_uids"]),vector(data[k]["class_trials"]),
                data[k]["subject_count"],vector(f"{x:.2f}" for x in data[k]["words_by_class"])] for k in ("train","val")]),
        "每批按类别优先取累计曝光最少的 21 个不同 UID，随机打破平局。每 UID 只取一条 trial，优先轮换低曝光 trial，遇到并列再平衡该类当前 batch 及累计的被试次数。每个种子先生成完整的有序索引，两组逐批读取同一数组，并从同一参数副本开始。每个 arm 共 3,276 个训练位置，三类各 1,092 个。",
        table(["种子","覆盖 UID","不同 trial","每 UID 曝光范围 N/U/P","采样平均词数 N/U/P","有序 batch SHA256 前 16 位"],audit),
        "运行时对全部 156 个独立 batch 逐一断言 UID 唯一且三类各 21 个。每 anchor 有 20 个合法正样本、42 个合法负样本；每批有向正/负对为 1,260/2,646，占 32.26% / 67.74%。没有无效 anchor。实现跳过无正样本的 anchor，若全批均无正样本则返回可反传的零损失，该分支已在 smoke 中检查。",
        f"首个真实 batch 不执行优化器更新。CE {fmt(smoke['ce'])}，SupCon {fmt(smoke['supcon'])}，组合损失 {fmt(smoke['total'])}；单独 SupCon 对投影权重的梯度范数 {fmt(smoke['supcon_project_grad'])}。组合损失对投影和分类器的各组参数均产生有限非零梯度，正式 312 次更新均通过数值检查。",
        "被试曝光次数按类别的明细、初始参数和有序采样哈希见 [sampling.json](sampling.json)。每类每位被试的 trial 可用性不同，无法完全平衡。全体样本属于同一任务；长度只做分层敏感性检查。词汇、题材、个体 EEG 质量及被试×句子交互仍可能影响结果。各缓存及 checkpoint 的完整哈希见 [preflight.json](preflight.json)。"])
    classification=[]
    matrices=[]
    for s,k,v in rows:
        y=v["val"]
        classification.append([s,k,fmt(v["train"]["f1"]),fmt(y["f1"]),fmt(y["ba"]),vector(map(fmt,y["recall"])),
                               vector(map(pct,y["pred_share"])),fmt(v["gap_f1"]),fmt(v["gap_ba"])])
        matrices.append([s,k,*[vector(t) for t in y["cm"]]])
    mean={}
    for k in ARMS:
        mean[k]={metric:float(np.mean([r[s][k]["val"][metric] for s in SEEDS])) for metric in ("f1","ba")}
    df=mean["S1"]["f1"]-mean["S0"]["f1"]
    db=mean["S1"]["ba"]-mean["S0"]["ba"]
    markdown("RESULTS.md",[
        "# 跨 UID 情感分类",
        "每个 UID 先对各 trial 的 softmax 概率取均值，再对三类取 argmax。40 个验证 UID 中 Negative/Neutral/Positive 分别有 8/16/16 个。下表 N/U/P 均按此次序。gap 为训练减验证。各组训练 Balanced Accuracy 均为 1.0000。",
        table(["种子","方法","训练 F1","验证 F1","验证 BA","验证 Recall N/U/P","验证预测比例 N/U/P","F1 gap","BA gap"],classification),
        "验证混淆矩阵，行是真实类别，行内为预测 N/U/P 的 UID 数。",
        table(["种子","方法","真实 N","真实 U","真实 P"],matrices),
        f"三种子等权均值：Macro-F1 {fmt(mean['S0']['f1'])} → {fmt(mean['S1']['f1'])}，差 {df:+.4f}；Balanced Accuracy {fmt(mean['S0']['ba'])} → {fmt(mean['S1']['ba'])}，差 {db:+.4f}。以验证 UID 为单位、按真实类别分层做 5,000 次配对 bootstrap，三种子的平均 S1−S0 差值 95% 描述区间分别为 [{ci[0,0]:+.4f}, {ci[0,1]:+.4f}]、[{ci[1,0]:+.4f}, {ci[1,1]:+.4f}]。这些区间以固定三个种子、当前类别构成和验证集为条件。",
        "42、43 两个种子，两组矩阵相同。种子 44 的微幅提升伴随 Positive 预测由 37 增至 39；Negative 与 Neutral 的 recall 仍为零。所有种子的训练 F1 都达到 1。验证证据没有显示新增监督稳定改善泛化。",
        "逐 UID 概率在 uid_种子_方法_val.csv；每步 CE、SupCon 和裁剪前总梯度范数在 loss_种子_方法.json。"])
    geom=[]
    for s,k,v in rows:
        e=v["embedding"]
        for q,direction in [("train_train","训练→训练"),("val_train","验证→训练"),("val_val","验证→验证")]:
            g=e[q]
            geom.append([s,k,direction,fmt(g["same"]["mean"]),vector(map(fmt,g["same"]["q10_q50_q90"])),
                         fmt(g["different"]["mean"]),vector(map(fmt,g["different"]["q10_q50_q90"])),
                         fmt(g["separation"])])
    retrieval=[]
    for s,k,v in rows:
        e=v["embedding"]
        retrieval.append([s,k,pct(e["nearest"]),e["unique_donors"],pct(e["perm_mean"]),
                          vector(map(pct,e["perm_q025_q975"])),fmt(e["perm_p_ge"]),
                          pct(e["length_matched"]),vector(map(pct,e["length_and_disjoint_subjects"]))])
    markdown("EMBEDDING_ANALYSIS.md",[
        "# 跨 UID 投影空间与检索",
        "每条 trial 取 256 维单位投影 z，再按 UID 均值并归一化为句子中心。对每个 anchor 分别汇总其他同标签和异标签 UID 的余弦，相同 UID 始终排除。均值与分位数均按 anchor 等权，每个 anchor 内合资格配对均分其权重。以下每行给出均值、10/50/90 分位数和均值差。配对之间共享 UID，配对数不能当成独立样本量。",
        table(["种子","方法","方向","同情感均值","同情感 P10/50/90","异情感均值","异情感 P10/50/90","同−异"],geom),
        "训练 UID 间类别分离很强，三个 S1 的训练同−异均略大于 S0。验证→训练在两组均观察到正的同−异均值差，SupCon 的增量方向随种子变化。仅在未见的验证句子之间计算时，六组均值差均不到 0.006，同/异类分布高度重叠。训练形成的分离未表现为稳定的新句情感聚集增益。",
        "## 验证→训练 1-NN",
        "每个验证 UID 在 320 个不同训练 UID 中按余弦取最近邻；同分按 UID 顺序。随机类频率基线为两划分类频率内积 33.5625%。训练多数类 Positive 在验证的命中率为 40.0%。随机标签对照保留具体供体和训练 UID 的类别数量，将训练 UID 标签整体置换 5,000 次，查询真实标签固定；p 为单侧加一校正的置换尾概率。几条查询若命中同一供体，其标签随同一置换改变。",
        table(["种子","方法","1-NN","不同供体","置换均值","置换 95% 范围","p","同句长箱","同句长箱＋异被试组 两方向"],retrieval),
        "六组均为 16/40 命中。其中 39/40 个查询的最近邻标签是 Positive。40% 与多数类基线相同，且在每组随机标签对照的范围内。限制查询和供体句长在同一训练三分位箱（切点 13、21 个空白分词）时仍均为 40%。",
        "另一敏感性检查按被试 ID 字典序交替分成两组，用一组被试构造查询 UID 中心、另一组构造训练供体中心，再交换方向，同时匹配句长箱。六组结果均为 37.5% / 45.0%。这是检索阶段的被试隔离检查，训练 probe 时仍使用全部训练被试；不能据此推断未见被试泛化。",
        "逐 UID 供体见 retrieval_种子_方法.csv。centroids_种子_方法.npz 保存可重算余弦与最近邻的 UID 中心；[results.json](results.json) 包含投影空间各组有效 anchor 数和有向配对数。"])
    markdown("SUMMARY.md",[
        "# SupCon-01 结论",
        "**在本次冻结 V3 特征的轻量 Probe 中，SupCon 没有稳定提高未见句子的情感恢复，也没有缓解 Positive 偏置。建议按停止标准暂缓接入生成或扩大情感专家。**",
        table(["问题","本次证据"],[
          ["跨 UID 泛化",f"三种子平均验证 Macro-F1：CE {fmt(mean['S0']['f1'])}，CE＋SupCon {fmt(mean['S1']['f1'])}；Balanced Accuracy：{fmt(mean['S0']['ba'])}、{fmt(mean['S1']['ba'])}。"],
          ["Positive 偏置","种子 42、43 均保持 36/40 个 Positive。种子 44 从 37/40 增到 39/40，Negative 和 Neutral 各无一个判对。"],
          ["跨句情感结构","训练句之间的同/异类分离略增。未见验证句之间的同−异余弦差均低于 0.006。六组验证→训练最近邻一致率均为 40%，等同多数类基线。"],
          ["三个种子","42、43 的两项验证分类指标各自不变；44 的微弱提升只伴随 Positive recall 增加。近邻命中数三个种子均无提升。"],
          ["进入生成阶段","依据本轮验证证据暂缓。当前比较检验冻结 V3 特征、此投影、固定 λ/τ 与预算的组合；未测生成质量或其他对比任务。"]]),
        "三组配对均使用相同初始参数和 batch，每组 52 步。CPU 完成数据审计、单批 smoke 及全部 6 组训练。复用 3,316 个训练 trial（320 UID）与 406 个验证 trial（40 UID），未运行 V3 或 Decoder，也未评估 Test。训练数据的高可分性和验证失败之间的成因仍未识别。",
        "详情见 [CONFIG.md](CONFIG.md)、[BATCH_SAMPLING_AUDIT.md](BATCH_SAMPLING_AUDIT.md)、[RESULTS.md](RESULTS.md) 与 [EMBEDDING_ANALYSIS.md](EMBEDDING_ANALYSIS.md)。"])
    print("Verified six pairs of UID split confusion matrices and six validation retrievals. Reports generated.")


if __name__=="__main__":
    main()
