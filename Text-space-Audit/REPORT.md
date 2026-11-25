# Text-space Audit：ZuCo1 Task 1 文本情感空间

**完成时间：2026-11-07（北京时间）**  
**统计单位：400 个独立句子；没有使用 EEG 特征。**  
**主模型：V3-Long 完整 epoch 46 checkpoint；发布版 GLIM checkpoint 仅作文本 Aligner 参照。**

## 1. 数据与权重核验

E1 `run-02/split_manifest.csv`（SHA256 `13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`）与现有 ZuCo1 Task 1 标签参考表（SHA256 `417a9bc980cfb40b448c09806cf3d38b8aa88e77f6393fa550f5a1ac1544ea41`）按 sentence UID 一对一核对：UID、规范化输入文本和三分类标签全部一致。400 句均有唯一 UID 和唯一规范化文本；Train/Val/Test 为 **320/40/40** 句，跨划分 UID 与规范化文本重叠均为 **0**。官方三类计数：negative 123、neutral 137、positive 140；训练集分别 103/104/113，验证集 8/16/16，测试集 12/17/11。保留了标签参考表的 `raw label`。没有读取 14GB EEG pickle，没有重新划分或创建标签。

V3 权重 `/root/autodl-tmp/A3-v2-V3-50-A800/run-02/checkpoints/epoch=46-step=93436.ckpt`，SHA256 `fe30e8cfc308aa464caa6d460b82bd1720c857d029c9944758b9d0661884c337`。发布版 GLIM 权重 SHA256 `25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`。两次使用同一份本地 Flan-T5-Large `model.safetensors`，SHA256 `a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e`，其 Encoder 在两次提取的 masked-mean 输出**逐值相同**（最大绝对差 0）。文本模型和全部 GLIM 参数均 `eval()`、冻结、无梯度；没有下载权重。

## 2. GLIM 实际对齐文本向量

`model/glim.py` 的 `encode_text()` 将 `input text` 用 Flan-T5-Large Encoder 编码，取最后一层 `last_hidden_state`；最大长度 96，padding 由 tokenizer attention mask 标记。`model/modules.py` 的 `Aligner.embed_text()` 用可学习文本 query `q_y` 对最后一层 token 表示做 **带 mask 的 cross-attention pooling**，输出每句 1024 维向量。`Aligner.align_emb_vector()` 对文本向量与 EEG 向量分别 L2 归一化，做双向 CLIP 交叉熵。这份**归一化后的 `q_y` 文本向量**是本报告主分析对象 `clip_text`。模型在训练与验证中调用同一个编码和对齐函数；本审计统一使用 `eval()` 和相同的 BF16 推理路径。

补充对象 `encoder_mean`：对同一最后层 token 表示按 tokenizer mask 做 mean pooling，再 L2 归一化，维度亦为 1024。它**不经过 GLIM 的文本 query，也不是原 CLIP 损失直接使用的向量**；不是另一层文本 Encoder。其他 Encoder 隐层未提取。

两个对象的单批 2 句 smoke test 均为 `[2,1024]`、数值有限；完整提取均为 `[400,1024]`。文本 token 长度 7–68，平均 27.54。三类 train 原型分别是 `Normalize(Mean(train class embeddings))`，存储于 `run-01/*_prototypes.npy` 和对应 CSV；类别列顺序为 negative、neutral、positive，均为 1024 维单位向量。Val/Test **未参与原型、标准化器、PCA 或探针拟合**。

## 3. 原型相似度与余弦空间结构

V3 主空间 `clip_text` 的训练集原型余弦相似度（类序 negative、neutral、positive）：

| | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1.000 | 0.120 | 0.421 |
| Neutral | 0.120 | 1.000 | 0.834 |
| Positive | 0.421 | 0.834 | 1.000 |

补充 `encoder_mean` 的三组原型相似度依次为 **N−U 0.975、N−P 0.921、U−P 0.956**，表明这些原型角度差很小。其测试集原型预测的 top-1/top-2 平均余弦 margin 仅 **0.022**；V3 主空间为 **0.202**。

| 表征 | Train 类内距离 N/U/P | Train 类间距离 N−U/N−P/U−P | 加权类内/类间 | Cosine silhouette Train / Val / Test |
|---|---|---|---:|---|
| V3 `clip_text`（主） | 0.209 / 0.381 / 0.188 | 0.915 / 0.662 / 0.406 | 0.391 | 0.393 / 0.079 / 0.075 |
| V3 `encoder_mean`（补充） | 0.411 / 0.457 / 0.400 | 0.445 / 0.449 / 0.450 | 0.941 | 0.042 / 0.011 / −0.002 |
| 发布版 GLIM `clip_text`（参照） | 0.254 / 0.341 / 0.315 | 0.302 / 0.290 / 0.326 | 0.995 | −0.046 / −0.059 / −0.157 |

V3 主空间在训练句子上的几何分离很强，但验证/测试 silhouette 明显降低，且训练集 Neutral 的 silhouette 仅 **0.016**；测试集 Neutral 为 **−0.420**。因此不能将训练集散点图中的聚集直接解释为稳定的三类原型结构。发布版 GLIM 三类原型几乎重合（N−U 0.991、N−P 0.990、U−P 0.999），且主空间测试 silhouette 为负。V3 与发布版训练步数和附加任务不同，这只是描述性参照，不能把空间变化单独归因于情感监督。

## 4. 固定原型分类与线性探针

每个方法只用 320 个 train 句拟合。原型分类以余弦最大值预测，不用验证标签更新原型。L2 多项逻辑回归仅用 train 拟合标准化器与权重，在 Val 从 C=`[0.01,0.1,1,10]` 选择正则强度（相同分数取较小 C）；Test 只用于最终评价。置信区间为 2,000 次句子级 bootstrap，40 句测试集仍有较宽不确定性。

| 权重/表征 | 方法 | Val Macro-F1 | Test Macro-F1 [95% CI] | Test Balanced Acc. | Test 各类 F1 N/U/P |
|---|---|---:|---:|---:|---|
| V3 `clip_text`（主） | Train 原型 | 0.454 | 0.425 [0.277, 0.560] | 0.463 | 0.615 / 0.231 / 0.429 |
| V3 `clip_text`（主） | 线性探针，C=0.1 | 0.550 | 0.623 [0.452, 0.775] | 0.654 | 0.609 / 0.571 / 0.690 |
| V3 `encoder_mean`（补充） | Train 原型 | 0.628 | 0.821 [0.680, 0.927] | 0.822 | 0.737 / 0.811 / 0.917 |
| V3 `encoder_mean`（补充） | 线性探针，C=0.1 | 0.641 | 0.706 [0.546, 0.830] | 0.713 | 0.636 / 0.647 / 0.833 |
| 发布版 GLIM `clip_text`（参照） | Train 原型 | 0.323 | 0.241 [0.121, 0.359] | 0.302 | 0.444 / 0.087 / 0.190 |
| 发布版 GLIM `clip_text`（参照） | 线性探针，C=10 | 0.439 | 0.583 [0.417, 0.726] | 0.602 | 0.667 / 0.467 / 0.615 |
| Train 多数类 Positive | 恒定预测 | — | 0.144 | 0.333 | 0 / 0 / 0.431 |

V3 主空间测试混淆矩阵（行是真实 N/U/P，列是预测 N/U/P）：

| 方法 | N 行 | U 行 | P 行 |
|---|---|---|---|
| 原型 | `[8,1,3]` | `[6,3,8]` | `[0,5,6]` |
| 线性探针 | `[7,2,3]` | `[4,8,5]` | `[0,1,10]` |

补充 `encoder_mean` 测试混淆矩阵：原型为 `[[7,5,0],[0,15,2],[0,0,11]]`，线性探针为 `[[7,5,0],[3,11,3],[0,1,10]]`。各预测类别数、Precision、Recall、每句余弦相似度和探针概率均保存在 `run-01/results.json` 与 `run-01/*_predictions.csv`。

## 5. 稳健性与辅助控制

将 **train 标签**随机置换 1,000 次，每次重新计算三个原型而不改变 Val/Test：V3 主空间的 Val/Test 原型 Macro-F1 经验单侧 p 分别约 **0.127/0.311**；补充 `encoder_mean` 为 **0.003/0.001**。这是探索性置换对照，不能据此在测试集挑选空间。仅用文本 token 数、词数和字符数训练同协议线性基线，Val/Test Macro-F1 为 **0.302/0.365**，低于上述主要文本向量探针。V3 主空间 PCA 第一/第二主成分解释 train 方差 **59.6%/13.3%**，与 token 长度 Spearman 相关约 **0.080/0.056**；发布版 GLIM 主空间 PC1 与长度相关 **−0.763**。这些控制排除“只有句长”这一简单解释，但不能排除词汇、数据来源或模型预训练语料带来的其他线索。

相同 40 测试句子的配对 bootstrap：V3 主空间原型相对发布版 GLIM 原型 Macro-F1 高 **0.184**，95% CI **[−0.007, 0.370]**；线性探针高 **0.040**，CI **[−0.175, 0.254]**。V3 `encoder_mean` 原型相对 V3 主空间原型高 **0.397**，CI **[0.231, 0.561]**；但该补充表征不是 GLIM 原 CLIP 空间，且这个比较属于本次探索性分析。

PCA 仅在 train 拟合并展示全部 400 句，见 `run-01/pca_clip_text.png` 和坐标 CSV。环境没有 `umap-learn`，因此 **UMAP 未运行**，没有安装新依赖。其他 T5 隐层未提取。

## 6. 判断与后续建议

**文本 Encoder 确实包含可读的情感信息。** 尤其最后层 masked mean 表征，在预先固定的 train 原型下对 40 句测试集取得较高 Macro-F1，且 train 标签置换对照较有力。**GLIM 当前直接用于 EEG−文本 CLIP 对齐的 V3 `q_y` 向量，不足以作为稳定的三类固定情感原型**：Train/Val/Test silhouette 不一致、Neutral 不稳定，原型分类的测试结果未明显优于置换原型。

因此建议**不要直接把 V3 `clip_text` 三个中心当作 A3.1 的情感原型目标**。如果继续 Emotion Prototype Alignment，可先把冻结 Flan-T5 `encoder_mean` 原型作为候选的独立辅助文本目标，明确设计与 EEG 分支之间的投影并在预注册的未见句子/真实 EEG 对照中验证；原 GLIM 文本生成与 CLIP 路径保持原样。还需在另一独立文本集合和更多句子上检验原型稳定性、近角度 margin、词汇/来源偏差，以及发布版 Flan-T5 预训练语料是否包含当前句子。文本-only 结果**不能证明 EEG 已携带同样的情感信息**。

## 7. 文件、完整性与限制

- V3 主实验：`/root/autodl-tmp/Text-space-Audit/run-01/`；发布版 GLIM 参照：`/root/autodl-tmp/Text-space-Audit/run-02-released-glim/`。
- 每个目录含 `audit.json`、`smoke.json`、`results.json`、`geometry_by_split.json`、`exploratory_controls.json`、`text_embeddings.npz`、三类原型 CSV/NPY、逐句预测 CSV、PCA 图和坐标 CSV。
- 根目录 `comparison.json` 含配对差异。`run-01.log` 与 `run-02-released-glim.log` 保留执行日志。`artifact_sha256.json` 与本报告的 SHA256 清单用于完整性核验。
- 未修改原始训练代码、checkpoint 或 E1/E1.1 文件；未启动 EEG 训练。结论局限于 ZuCo Task 1 的 400 句与两个现有 checkpoint；40 句独立测试样本较小，且 Flan-T5 预训练语料逐句溯源不可得。
