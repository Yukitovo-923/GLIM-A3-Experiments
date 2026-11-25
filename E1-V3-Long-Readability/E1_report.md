# E1：V3-Long 情感线性可读性（探索性）

## 权重与方法

- 权重：`/root/autodl-tmp/A3-v2-V3-50-A800/run-02/checkpoints/epoch=46-step=93436.ckpt`，SHA256 `fe30e8cfc308aa464caa6d460b82bd1720c857d029c9944758b9d0661884c337`。完整 epoch 46；epoch 47 曾触发梯度保护。
- 执行源码哈希：`model/vq.py` 为 `a443ab7a60062dd6e969337bff612616b82f99c1044fe23a534f033c4603adc2`；`model/glim_multitask.py` 为 `74e930c3d51afeb912e0e63b4e9f13fdbecb4ff18bc57620e0f05ce157a58000`。
- 模型全部冻结、`eval()`、无梯度；GPU 仅提取特征，SciPy CPU L-BFGS-B 拟合 L2 多项逻辑回归。每种特征使用相同 C 网格 `[0.01,0.1,1,10]`；仅验证集选 C。标准化器仅由 train 拟合。按句子 UID 给 trial 反频数权重，验证/测试先将同一句不同被试的预测概率平均，再计算指标。
- 池化：96 个有效 EEG query 的固定均值。实际维度：Encoder 256、Aligner `in_proj` 1024、VQ 量化向量 256。VQ 与 Aligner 是并行路径，并非 Aligner 后再 VQ。

## 划分审计

ZuCo1 Task 1 共 400 个带三分类标签的独立句子、4,139 条 EEG trial。Train 320 句/3,316 trial，Val 40 句/406 trial，Test 40 句/417 trial。Train 的 negative/neutral/positive trial 为 1094/1069/1153，Val 为 78/165/163，Test 为 121/181/115。保存的 manifest 中 UID 和规范化文本跨划分重叠均为零，同一 UID 标签冲突为零。V3 spherical K-means 初始化抽取的 953 个 UID 全属 train，无 val/test UID。

**限制：** 发布 GLIM 原始 checkpoint 没有逐句训练 UID 清单，无法单靠 checkpoint 证明其预训练绝未接触当前 val/test 句子。本报告因此标为探索性，不宣称已完全证明预训练层面的未见句子泛化。

## 独立句子测试结果

| 表征 | 验证 Macro-F1 | 测试 Macro-F1 | 测试 95% CI | Balanced Acc. | 多数类基线 |
|---|---:|---:|---:|---:|---:|
| Encoder 池化 | 0.194 | 0.144 | [0.087, 0.199] | 0.333 | 0.144 |
| Aligner 连续 | 0.194 | 0.144 | [0.087, 0.199] | 0.333 | 0.144 |
| VQ 量化 | 0.194 | 0.150 | [0.089, 0.202] | 0.333 | 0.144 |

三类测试句子数：negative 12、neutral 17、positive 11。Encoder 与 Aligner 的测试预测全部落到 positive；VQ 也几乎全部落到 positive。详细混淆矩阵、各类 F1、打乱标签基线和置信区间见 `run-02/report_with_metric_ci.json`。打乱标签探针在这个 40 句测试集上偶尔高于真实标签探针，进一步表明结果不稳定。

## 固定探针的测试集控制对照

以下为 seed 42 的 Macro-F1。差值 = Real EEG − Control，95% CI 按独立句子配对 bootstrap。

| 表征 | Real | 分布匹配 Gaussian | Gaussian 差值 | Temporal Shuffle | Paired EEG Shuffle |
|---|---:|---:|---:|---:|---:|
| Encoder | 0.144 | 0.285 | −0.141 [−0.268, −0.017] | 0.184 | 0.147 |
| Aligner | 0.144 | 0.271 | −0.127 [−0.277, +0.025] | 0.184 | 0.147 |
| VQ | 0.150 | 0.328 | −0.178 [−0.329, −0.017] | 0.187 | 0.147 |

Gaussian 用 train 的 104 个真实 EEG 通道各自均值和标准差拟合；有效时间位置替换，时间 padding 和 24 个补零通道保留。Temporal Shuffle 仅重排有效时间位置；Paired EEG Shuffle 在同一被试、不同句子之间错配。随机种子 42/43/44 下，Real−Gaussian 的三种表征差值全部为负；完整值见 `run-02/controls_multiseed.json`。这说明当前固定探针未显示真实 EEG 相对这些控制的稳定优势；控制输入属于分布干预，不应据此推断噪声“携带情感”。

## 码本、Query 与被试

Task 1 train/val/test 的活跃 code 分别为 33/91/60，不能用全任务的 124/128 活跃码替代 Task 1 的统计。训练集 code ID 与 96 个 query 位置的互信息为 0.876 bit，占 code 熵的 22.9%；测试集为 0.620 bit，占 15.7%。位置因素明显，码使用率不等于情感语义。测试集特征的被试间方差比例约 4–5%，仅作描述；详细值见 `run-02/confounds.json`。

## 判断

当前 E1 **不支持以“V3 离散特征已稳定编码情感”为理由推进 A3.1 双码本**。如果继续 A3.1，应定位为需要重新验证的探索性设计，并先补齐发布 checkpoint 的训练 UID 溯源、扩大独立句子测试以及处理 query 位置和分布偏移。首轮 `run-01` 因探针 L2 缩放问题仅保留审计；本报告采用修正后、独立保存的 `run-02`。
