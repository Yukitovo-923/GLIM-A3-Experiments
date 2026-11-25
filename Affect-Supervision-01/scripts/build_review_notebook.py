"""Build and execute a self-contained, data-backed review notebook.

Only reads artifacts within Affect-Supervision-01. Requires numpy and matplotlib.
"""

import base64
import contextlib
import io
import json
import os
import textwrap
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "experiment_review.ipynb"
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(source).strip() + "\n"})


md("""
# Affect-Supervision-01｜实验流程与设计审查

**问题：**在 V3 GLIM 已有连续 EEG–文本对齐及情感 CE 的基础上，新增 EPA 原型监督是否改善 EEG-to-Text？  
**状态：**B0/B1 × 种子 42/43/44 已训练、测试并评分。图表均从本实验保存的 JSON、CSV 和预测文件计算；重新运行 notebook 即可复核。  
**读法：**按实际执行顺序查看。每节末尾的“检查点”指出可能改变结论的设计选择或漏洞。

本 notebook 只分析已有结果，不读取旧实验目录、不启动训练、不调整测试集超参数。运行目录可设为实验根目录或其子目录。依赖：Python、NumPy、Matplotlib。
""")

code("""
from pathlib import Path
import csv, json, math, collections, hashlib
import numpy as np
import matplotlib.pyplot as plt

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'configs/common.json').is_file()), None)
assert ROOT is not None, 'Run from Affect-Supervision-01 or one of its subdirectories'
read = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8'))
rows = lambda p: list(csv.DictReader((ROOT / p).open(encoding='utf-8-sig', newline='')))
cfg = read('configs/common.json')
audit = read('reports/data_audit.json')
fair = read('reports/fairness_audit.json')
sha = read('reports/sha256_audit.json')
train = read('reports/training_summary.json')
comparison = read('evaluation/comparison.json')
status = read('evaluation/evaluation_status.json')
assert status['status'] == 'complete' and len(train['runs']) == 6
print('Experiment:', ROOT.name)
print('Evaluation completed:', status['ended_at'])
print('Source checkpoint SHA256:', sha['remote_checkpoint']['sha256'])
print('Frozen prototype SHA256:', sha['remote_assets']['train_only_encoder_mean_prototypes_sha256'])
""")

md("""
## 1. 审计与边界设定

1. **原模型兼容性：**指定 V3 epoch 46 checkpoint 与 `A3-full` 严格匹配，215 个共享 state tensor；V3 已有离散情感头、VQ 和 Attribute Token。EPA-01 原实现需要 `A3-no-attribute`，不能直接作为本轮 B0/B1 的模型。
2. **实验含义：**方案 B 保留原离散情感 CE `0.2`，B1 额外加入 EPA CE `0.2`。因此比较的是“额外 EPA”的增量效应，不能解释为“有情感监督 vs 无情感监督”。
3. **隔离：**原 checkpoint、EEG 数据、V3/EPA-01/Topic-EPA 目录只读。所有新代码、日志和 checkpoint 位于本实验目录。
4. **运行环境：**正式训练前检查 A800 80GB、GPU 进程、磁盘和 checkpoint；记录为 PyTorch 2.3.0+cu121、CUDA runtime 12.1、Lightning 2.4.0，BF16 可用。训练按配对种子顺序执行，未终止其他任务。

**检查点：**如果目标其实是“情感监督是否总体有益”，需增加真正无情感监督的对照；那将是另一个实验，不能把当前 B0 改名代替。
""")

code("""
assert fair['shared_weights_elementwise_equal'] and fair['shared_v3_state_dict_keys'] == 215
assert cfg['loss']['discrete_emotion_weight'] == cfg['loss']['epa_weight_b1_only'] == 0.2
assert sha['remote_checkpoint']['sha256'] == cfg['checkpoint']['sha256']
print('Shared V3 weights:', fair['shared_v3_state_dict_keys'], 'identical tensors')
print('Base losses equal on the real smoke batch:', ', '.join(fair['shared_base_loss_terms_equal_on_real_batch']))
print('Initial generated text equal:', fair['initial_generated_text_equal'])
print('Design: B0 = V3 objective; B1 = V3 objective + 0.2 × EPA CE')
""")

md("""
## 2. 数据、标签与 UID 划分

ZuCo1 Task1-SR：400 个 UID、4,139 个有效 EEG trial、12 名被试。按 UID 的 train/val/test 为 320/40/40 句，trial 为 3,316/406/417；同句不同被试 trial 同属一个 split。原 EPA-01 的 UID 划分可复用，但其跨任务训练 loader 不适合本轮 Task1 专项实验。训练仍沿用 V3 的每 trial 8 种文本变体，验证/测试使用原始句。

**检查点：**被试 trial 和句子不是独立样本。主生成指标先对 UID 内 trial 取均值，再对 40 个 UID 等权平均；不能把 417 trial 当作 417 个独立测试句。标签描述文本情感倾向，不是被试真实情绪。
""")

code("""
manifest = rows('data_manifests/zuco1_task1_sr_uid_split.csv')
uid_phase = collections.defaultdict(set)
uid_label = collections.defaultdict(set)
for r in manifest:
    uid_phase[r['text uid']].add(r['phase'])
    uid_label[r['text uid']].add(r['sentiment label'])
assert len(uid_phase) == 400 and all(len(v) == 1 for v in uid_phase.values())
assert all(len(v) == 1 for v in uid_label.values())
assert audit['invalid_trial_count'] == 0
splits = ['train','val','test']
classes = ['negative','neutral','positive']
fig, ax = plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
ax[0].bar(splits,[audit['sentence_split_counts'][s] for s in splits],color='#457b9d')
ax[0].set_title('Independent sentence UIDs'); ax[0].set_ylabel('Count')
ax[1].bar(splits,[audit['valid_trial_split_counts'][s] for s in splits],color='#2a9d8f')
ax[1].set_title('Valid EEG trials')
ax[2].bar(classes,[audit['sentence_label_counts'][c] for c in classes],color=['#d65a57','#9aa0a6','#e2a346'])
ax[2].set_title('Sentence labels'); ax[2].tick_params(axis='x',rotation=20)
plt.show()
print('UID split overlap:', audit['uid_split_overlap'], '| label conflicts:', audit['uid_label_conflict'])
print('Subjects:', audit['subject_valid_trial_counts'])
""")

md("""
## 3. 文本原型与模型路径

训练集 320 UID 的文本经冻结 Flan-T5-Large Encoder（`To English:` 前缀、masked mean）得到句向量；句向量 L2 归一化，按 Negative/Neutral/Positive 分组取均值后再次归一化，形成冻结 `[3,1024]` 原型。测试句没有进入原型计算，原型 SHA256 见首节。

```text
EEG → V3 EEG Encoder → 连续对齐 / q_x → 原 V3 生成 memory → 冻结 Flan-T5 Decoder
                               └─ B1: Linear(1024,1024) → L2 → cosine(3 frozen prototypes)/0.07 → CE
V3 离散情感头 → Attribute Token ──────────────────────────────────────┘（B0/B1 均保留）
```

**检查点：**EPA 只给 EEG 侧 `q_x` 梯度；其原型相似度并不直接作为 Decoder 输入。若情感恢复提高但生成不变，可能是监督未有效影响生成 memory。这里的原型与 GLIM 原 `q_y` 文本对齐空间并非同一目标。

推理实现为计算 LM Loss 仍调用完整 `model(batch)`，因此会编码参考文本；代码路径中，eval 模式下文本表征只参与对齐损失，生成 memory 来自 EEG，文本 mask 不用于生成 memory。为了把这个静态审计变为实证，下一轮应固定 EEG/提示、替换 `input text`，逐值验证 `eeg_embeds`、`memory_mask` 和生成 token 完全不变。
""")

code("""
proto = read('evaluation/bertscore_model.json')
print('Text prototype shape: [3,1024]; classes:', classes)
print('EPA temperature:', cfg['loss']['epa_temperature'])
print('Text model frozen and V3 Attribute Token present in smoke test:',
      all(x['attribute_token_present'] for x in read('reports/smoke_report.json')['arms'].values()))
print('BERTScore evaluator (separate from prototype encoder):', proto['model'], proto['revision'])
""")

md("""
## 4. 公平训练、验证与 checkpoint 选择

两个 arm 使用相同 UID split、预处理、Adam、学习率 `1e-4`、batch 72、BF16、随机种子 42/43/44、最多 10 epoch。每轮按验证 LM Loss 保存 best/last；早停 patience 4。六个任务顺序在 A800 上执行，训练 2026-11-11 20:52–22:23。下图从逐 epoch 日志绘制。

**检查点：**六组均在第 5 个 epoch 早停，最佳都是 epoch 0。训练 LM Loss 与验证 LM Loss 的方向值得调查：学习率、已有 checkpoint 后继续训练的优化器重置、文本变体分布与验证原句的差异，都可能有关。测试集不能用于补救性调参。
""")

code("""
curve = rows('reports/training_curve.csv')
fig, ax = plt.subplots(2,3,figsize=(14,6.5),sharex=True,layout='constrained')
for j,seed in enumerate([42,43,44]):
    for arm,color in [('B0_Baseline','#457b9d'),('B1_EPA','#e76f51')]:
        z=sorted((r for r in curve if r['arm']==arm and int(r['seed'])==seed),key=lambda r:int(r['epoch']))
        ep=[int(r['epoch'])+1 for r in z]
        ax[0,j].plot(ep,[float(r['train_loss_lm']) for r in z],marker='o',label=arm,color=color)
        ax[1,j].plot(ep,[float(r['val_loss_lm']) for r in z],marker='o',label=arm,color=color)
    ax[0,j].set_title(f'Seed {seed}'); ax[1,j].set_xlabel('Completed epoch')
    ax[0,j].set_xticks([1,2,3,4,5]); ax[1,j].set_xticks([1,2,3,4,5])
ax[0,0].set_ylabel('Train LM loss'); ax[1,0].set_ylabel('Validation LM loss')
ax[0,2].legend(fontsize=8)
plt.show()
print([(r['arm'],r['seed'],r['epochs_completed'],r['best_epoch'],round(r['best_val_lm_loss'],4)) for r in train['runs']])
""")

md("""
## 5. 锁定测试评估：生成质量

六个最佳 checkpoint 先由验证集选定，再对同一 417 trial / 40 UID 测试集生成文本。解码配置统一：beam 2、最大 64 token。BLEU-1/2/4 与 ROUGE-L 使用 TorchMetrics；BERTScore F1 使用固定 RoBERTa-large 第 17 层，IDF 与 baseline rescale 均关闭。图中每条线连接**同一种子**的 B0/B1；平均值和标准差见 `comparison.json`。

首次测试推理发现原 V3 对齐层的无参 `.squeeze()` 在单样本尾批会去掉 batch 维。本实验仅在独立评估代码中把同被试的单样本尾批并入前一个 batch，模型参数和原 V3 代码未改；重跑后六组的正确/错配条件各保留完整 417 trial / 40 UID。

**检查点：**生成质量是主问题。不能因 ROUGE-L 单项均值略升就宣布成功；BLEU-2/4、BERTScore、LM Loss 和三种子方向也需一致。40 UID 的 bootstrap 区间只反映句子抽样，不替代更多种子的重复实验。
""")

code("""
metrics=['bleu1','bleu2','bleu4','rouge_l','bert_score_f1','lm_loss']
titles=['BLEU-1','BLEU-2','BLEU-4','ROUGE-L','BERTScore F1','LM loss (lower better)']
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for ax,key,title in zip(axes.flat,metrics,titles):
    for seed in [42,43,44]:
        a=comparison['per_seed']['B0_Baseline'][str(seed)]['matched'][key]
        b=comparison['per_seed']['B1_EPA'][str(seed)]['matched'][key]
        ax.plot([0,1],[a,b],marker='o',label=f'seed {seed}')
    ax.set_xticks([0,1],['B0','B1']); ax.set_title(title); ax.grid(alpha=.2)
axes[0,2].legend(fontsize=8)
plt.show()
for key in metrics:
    a=comparison['summary']['B0_Baseline']['matched'][key]
    b=comparison['summary']['B1_EPA']['matched'][key]
    d=comparison['paired_arm_delta_b1_minus_b0'][key]
    print(f'{key:15s} B0 {a["mean"]:.5f}±{a["sd"]:.5f} | B1 {b["mean"]:.5f}±{b["sd"]:.5f} | Δ {d["mean"]:+.5f}')
for key in ['bert_score_f1','rouge_l','bleu4']:
    ci=comparison['uid_bootstrap'][f'b1_minus_b0_{key}']['uid_bootstrap_95ci']
    print(key,'paired UID bootstrap 95% CI:',[round(v,5) for v in ci])
for arm in ['B0_Baseline','B1_EPA']:
    for seed in [42,43,44]:
        z=read(f'evaluation/outputs/{arm}_seed{seed}_test/inference_summary.json')
        assert all(z['conditions'][c]['trial_count']==417 and z['conditions'][c]['uid_count']==40
                   for c in ['matched','mismatched'])
print('Coverage check: all 12 condition outputs contain 417 trials and 40 UIDs.')
""")

md("""
## 6. 情感属性恢复

以每 UID 的被试 trial 概率均值进行句级分类。原 V3 情感头可用于 B0/B1 对比；B1 EPA 相似度可作为额外读出。下面的矩阵将三个种子的同 40 句结果**相加以显示预测模式**，并不把 120 次预测当作独立句子。每种子的精确率、召回率、F1、混淆矩阵保存在对应 `metrics.json`。

**检查点：**EPA 读出主要预测 Positive，Neutral 几乎完全失败。下一轮应先看原型角距离、logit 分布、类别权重和 EEG `q_x` 的范数/投影几何，再决定是否继续扩大离散分支。不能用测试集决定这些超参。
""")

code("""
heads=[('B0 V3 head','B0_Baseline','v3_discrete_head'),
       ('B1 V3 head','B1_EPA','v3_discrete_head'),('B1 EPA','B1_EPA','epa')]
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for axis,(title,arm,head) in zip(ax,heads):
    mats=[]
    for seed in [42,43,44]:
        m=read(f'evaluation/outputs/{arm}_seed{seed}_test/metrics.json')
        mats.append(np.array(m['conditions']['matched']['emotion'][head]['sentence']['confusion_matrix']))
    cm=sum(mats)
    axis.imshow(cm,cmap='Blues',vmin=0,vmax=50)
    axis.set_title(title); axis.set_xticks(range(3),['Neg','Neu','Pos']); axis.set_yticks(range(3),['Neg','Neu','Pos'])
    axis.set_xlabel('Predicted'); axis.set_ylabel('True')
    for i in range(3):
        for j in range(3):axis.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.show()
for arm in ['B0_Baseline','B1_EPA']:
    for key in ['v3_disc_sentence_macro_f1','v3_disc_sentence_balanced_accuracy','epa_sentence_macro_f1','epa_sentence_balanced_accuracy']:
        v=comparison['summary'][arm]['matched'].get(key)
        if v:print(arm,key,round(v['mean'],4),'+/-',round(v['sd'],4))
""")

md("""
## 7. EEG 利用程度：同被试错配

每条测试 trial 的 EEG 换成同被试、不同 UID 的 EEG，提示和参考句保持不变。下图使用每个种子的**正确配对减错配**差值：LM Loss 为负代表正确配对更好，生成指标为正代表正确配对更好。

**检查点：**六个模型的正确配对 LM Loss 都较低，表明条件似然依赖 EEG；然而 BERTScore 和 ROUGE-L 没有稳定的正确配对优势。这只支持“EEG 进入并影响模型”，尚不能证明生成文本可靠地恢复了 EEG 中的句子语义。
""")

code("""
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for ax,key in zip(axes,['lm_loss','bert_score_f1','rouge_l']):
    for arm,color in [('B0_Baseline','#457b9d'),('B1_EPA','#e76f51')]:
        vals=[comparison['per_seed'][arm][str(s)]['matched'][key]-
              comparison['per_seed'][arm][str(s)]['mismatched'][key] for s in [42,43,44]]
        offset=-.08 if arm=='B0_Baseline' else .08
        ax.scatter(np.arange(3)+offset,vals,label=arm,color=color)
        ax.plot(np.arange(3)+offset,vals,color=color,alpha=.4)
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(range(3),['42','43','44'])
    ax.set_xlabel('Seed'); ax.set_title(f'Matched − mismatched: {key}'); ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
plt.show()
for arm in ['B0_Baseline','B1_EPA']:
    print(arm,'mean matched−mismatched LM loss:',round(comparison['eeg_matched_minus_mismatched'][arm]['lm_loss']['mean'],4))
""")

md("""
## 8. 被试分布与定性生成

分被试表格来自完整测试预测。下图是每名被试在三个配对种子中 B1−B0 的 BERTScore F1 均值；不同被试读到的 UID 子集并不完全相同，因此被试差异不能单独解释为生理差异。示例预测只用于人工查看错误类型，不用于选择模型。

**检查点：**可以检查某些被试或句型是否特别受 EPA 影响，但任何基于测试集的改动都需另找验证集决定、再用封存数据复验。
""")

code("""
subject_rows=rows('evaluation/subject_results.csv')
subjects=sorted({r['subject'] for r in subject_rows})
deltas=[]
for subject in subjects:
    pair=[]
    for seed in [42,43,44]:
        find=lambda arm: next(float(r['bert_score_f1']) for r in subject_rows if r['arm']==arm and
                             r['subject']==subject and int(r['seed'])==seed and r['condition']=='matched')
        pair.append(find('B1_EPA')-find('B0_Baseline'))
    deltas.append(np.mean(pair))
fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
ax.bar(subjects,deltas,color=['#2a9d8f' if x>=0 else '#d65a57' for x in deltas])
ax.axhline(0,color='black',lw=.8); ax.set_ylabel('B1 − B0 BERTScore F1'); ax.set_title('Paired three-seed mean by subject')
plt.show()
for uid in [6,14,15]:
    out=[]
    for arm in ['B0_Baseline','B1_EPA']:
        p=ROOT/f'evaluation/outputs/{arm}_seed42_test/matched_predictions.jsonl'
        item=next(json.loads(line) for line in p.open(encoding='utf-8') if json.loads(line)['uid']==uid)
        out.append((arm,item['prediction']))
        ref=item['reference']
    print('UID',uid,'REF:',ref[:180])
    for arm,pred in out:print(arm+':',pred[:180])
""")

md("""
## 9. 结果判定与下一轮设计检查表

**本轮判定：**B1 的 BLEU-1 和 ROUGE-L 均值略高，BLEU-2、BLEU-4、BERTScore 均值较低；BERTScore 的 UID bootstrap 95% 区间跨 0。原 V3 情感头的句级 Macro-F1 从 B0 `0.22664` 降到 B1 `0.19214`，B1 EPA 读出 `0.16580`。B1 测试 LM Loss 从 `9.16890` 升至 `9.20136`，但种子间方向不一致。**不能宣称 EPA 改善生成或情感恢复，也不能断言整体语义对齐显著恶化。**

下一轮正式设计前建议按此顺序检查：

1. **优化器与早停：**为什么六组最佳都在 epoch 0？检查 continuation 学习率、优化器状态重置、训练 8 变体与验证原句分布。只用训练/验证数据重设方案。
2. **EPA 几何与类别塌缩：**在训练/验证集绘制三原型两两余弦、每类 logit 分布、预测类别比例、投影前后角距离，检查 `τ=0.07` 和 `λ=0.2` 的梯度规模。
3. **原 V3 情感监督上限：**当前 B0 已含情感头。若要测试离散情感监督是否有价值，需建立另一个初始化兼容、生成路径严格相同的无情感监督实验，而不是把本轮 B0 当作纯 GLIM。
4. **EEG 对文本的可识别作用：**继续用同被试错配，并加入多次错配、按 UID 置换置信区间和生成差异分析；先明确教师强制概率改善与自由生成质量之间的落差。
5. **推理文本依赖审计：**固定 EEG/提示并替换参考 `input text`，验证生成 memory 与文本输出不变；当前只有代码路径审计，没有运行时反事实记录。
6. **封存测试集：**先前 Text-space Audit 已诊断过这 40 个测试句，故本轮并非完全首见测试。后续修改在验证集完成，再用新的封存句子测试。

**源文件导航：**`reports/final_comparison.md`（正式报告）、`reports/training_curve.csv`（30 个 epoch 记录）、`reports/training_summary.json`（六组 checkpoint SHA256）、`evaluation/comparison.json`（均值/标准差/UID bootstrap）、`evaluation/outputs/*/metrics.json`（逐种子指标与混淆矩阵）、`evaluation/outputs/*/*_scored.jsonl`（逐 trial 预测）、`evaluation/subject_results.csv`（分被试）、`reports/sha256_audit.json`（源资产校验）。远端 checkpoint 在 `/root/autodl-tmp/Affect-Supervision-01/runs/`。
""")


def execute_notebook():
    namespace = {"__name__": "__main__"}
    count = 0
    os.chdir(ROOT)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        buffer = io.StringIO()
        outputs = []
        plt.close("all")
        with contextlib.redirect_stdout(buffer):
            exec(compile(cell["source"], f"notebook-cell-{count}", "exec"), namespace)
        if buffer.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": buffer.getvalue()})
        for figure_number in plt.get_fignums():
            fig = plt.figure(figure_number)
            data = io.BytesIO()
            fig.savefig(data, format="png", dpi=145, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(data.getvalue()).decode("ascii")}})
        plt.close("all")
        cell["execution_count"] = count
        cell["outputs"] = outputs


if __name__ == "__main__":
    execute_notebook()
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                 "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Built {OUT} with {len(cells)} cells and {sum(c['cell_type']=='code' for c in cells)} executed code cells")
