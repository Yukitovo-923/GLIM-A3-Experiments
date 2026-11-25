# Text-space Audit：`To English:` 前缀与无前缀对照

**完成日期：2026-11-07（北京时间）**  
**目的：修正文本输入协议审计，选择后续 EPA 的冻结文本情感原型候选。**

## 实验设置与完整性

保留既有无前缀 `run-01`（V3）和 `run-02-released-glim`（发布版 GLIM）及其脚本、结果、原型。本次新增 `run-03-v3-prefixed` 与 `run-04-released-prefixed`：仅将模型输入从原始句子改为 GLIM DataModule 实际使用的 `To English: <句子>`，其余协议完全沿用前次 Text-space Audit。文本模型、Aligner 和所有参数冻结，`eval()` 加无梯度推理；只使用 ZuCo1 Task 1 官方 400 句与原始三分类标签，没有 EEG 特征、训练、下载或重划分。

E1 `run-02/split_manifest.csv` 的 SHA256 为 `13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`；标签参考表 SHA256 为 `417a9bc980cfb40b448c09806cf3d38b8aa88e77f6393fa550f5a1ac1544ea41`。四组均对应 **400 个唯一 UID/规范化文本**，Train/Val/Test **320/40/40**，跨划分 UID 和规范化文本重叠 **0**；negative/neutral/positive 分别 **123/137/140**。V3 checkpoint SHA256 `fe30e8cfc308aa464caa6d460b82bd1720c857d029c9944758b9d0661884c337`；发布版 GLIM checkpoint SHA256 `25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`；共用 Flan-T5-Large 权重 SHA256 `a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e`。

带前缀 run 的 `audit.json` 明确记录 `text_input_template: "To English: <MASK>"` 及第一条实际模型输入。原始文本平均 token 数为 **27.54**；带前缀后 **30.54**，每句恰多 3 个 token。两组单批 smoke 均为 `[2,1024]`、数值有限；完整提取均为 `[400,1024]`。

## 主问题：`encoder_mean` 三类情感原型

两种输入均对 Flan-T5-Large 最后一层做 tokenizer-mask mean pooling、逐句 L2 归一化；**只用 320 个 train 句**分 negative 103、neutral 104、positive 113 求均值，再 L2 归一化成三个 1024 维单位原型。验证和测试均未参与原型拟合。每句的 raw 与 prefixed `encoder_mean` 余弦相似度平均 **0.829**；同类原型 raw/prefixed 余弦相似度分别 **0.837/0.831/0.838**，说明前缀明显改变向量，但并未使标签或句子对齐失效。

| 原型输入 | Val Macro-F1 [95% CI] | Test Macro-F1 [95% CI] | Test Balanced Acc. | Test 各类 F1 N/U/P |
|---|---:|---:|---:|---|
| 无前缀原始句子 | **0.628** [0.448, 0.773] | **0.821** [0.680, 0.927] | 0.822 | 0.737 / 0.811 / 0.917 |
| GLIM `To English:` 前缀 | 0.553 [0.386, 0.694] | 0.757 [0.603, 0.877] | 0.758 | 0.783 / 0.706 / 0.783 |

按相同 40 句的**配对** bootstrap（5,000 次），无前缀减带前缀的 Macro-F1：Val **+0.075**，95% CI **[−0.024,+0.194]**；Test **+0.064**，CI **[−0.062,+0.196]**。Val 仅 6 句、Test 仅 7 句改变预测类别；两个置信区间都包含 0。因此**按本轮声明的验证集规则暂选无前缀原型，但没有充分证据声称它稳定优于带前缀原型**；本轮没有用测试集重新选择。不过，此前 Text-space Audit 已公开无前缀方案的测试分数，故本次不是完全盲测，不能把现有 Test 当作新鲜独立确认。

原型测试混淆矩阵，行真实 N/U/P、列预测 N/U/P：无前缀 `[[7,5,0],[0,15,2],[0,0,11]]`；带前缀 `[[9,3,0],[2,12,3],[0,2,9]]`。无前缀更有利于本次 Test 的 Neutral 和 Positive，带前缀更有利于 Negative。详细 Precision/Recall、逐句预测和余弦分数保存在各 run 的 `results.json` 与 `encoder_mean_predictions.csv`。

三类原型相似度（N−U / N−P / U−P）：无前缀 **0.975/0.921/0.956**，带前缀 **0.981/0.944/0.968**。两者的原型都很接近；train 加权类内/类间余弦距离比为 **0.941/0.951**；cosine silhouette 的 Train/Val/Test 分别为 **0.042/0.011/−0.002** 与 **0.033/0.004/−0.004**。原型分类分数高不代表三类形成紧密且彼此远离的球形簇。训练标签置换 1,000 次的原型分类经验 p 值：无前缀 Val/Test **0.003/0.001**，带前缀 **0.008/0.001**；两种输入均显示标签相关可读性，但此分析是探索性检验。

## 线性探针对照

同一 L2 多项逻辑回归协议仅在 Train 拟合标准化器和权重，在 Val 的 C=`[0.01,0.1,1,10]` 中选择 C；两种 `encoder_mean` 均选 C=0.1。

| 输入 | Val Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| 无前缀 | 0.641 | 0.706 |
| 带前缀 | 0.590 | 0.706 |

两者 Test 线性探针几乎相同，配对差 **−0.001**，95% CI **[−0.135,+0.139]**。测试集的 train 多数类基线 Macro-F1 为 **0.144**；仅用 token 数、词数、字符数的线性基线为 **0.365**。此对照说明文本向量中的情感信息不是仅靠简单长度特征即可解释。

## GLIM 直接 CLIP 文本向量的补充审计

`clip_text` 是最后层经 GLIM `q_y` cross-attention pooling 后的归一化句级向量；它与 EPA 的 `encoder_mean` 原型空间不同。V3 无前缀/带前缀的原型 Val Macro-F1 为 **0.454/0.493**，Test 为 **0.425/0.432**；线性探针 Val 为 **0.550/0.526**，Test 均约 **0.623**。前缀没有明显改善未见句子分类。

特别需要注意：V3 带前缀 `clip_text` 的训练集 cosine silhouette 高达 **0.996**，但 Val/Test 仅 **−0.003/0.132**；类内平均距离约 **0.002**、类间约 **0.863**。PCA 图也显示三类训练句子挤成极紧密的小团，但未见句子广泛散布。脚本提取时没有将标签送入模型；这一现象仍需要作为**训练句子过拟合或表征退化线索**审查，不能拿训练集几何结构证明情感原型稳健。发布版 GLIM 的带前缀 `clip_text` Train/Val/Test silhouette 为 **−0.053/−0.079/−0.114**。两份 checkpoint 训练历史不同，不能把差异单独归因于情感监督。

两组 PCA 都仅在 Train 拟合，图见 `run-01/pca_clip_text.png` 与 `run-03-v3-prefixed/pca_clip_text.png`。当前环境未安装 UMAP，本轮未安装依赖，也未生成 UMAP 图。二维图只作辅助；判断以上述高维定量指标为准。

## 对 EPA 的选择

**本轮声明的选择指标是 V3 `encoder_mean` 原型在 Val 的句子级 Macro-F1，但这不是完全盲测的预注册选择，因为无前缀的 Test 结果此前已知。** 依此规则，后续 EPA 第一轮可把**无前缀原始句子**原型作为明确标注的“独立文本情感监督目标”候选，文件 `/root/autodl-tmp/Text-space-Audit/run-01/encoder_mean_prototypes.npy`，SHA256 `ceaaccee85eee260c8c6e540bf441fced6421b2912aa9014d246e0cd55f5f31d`。它**不是 GLIM 原训练输入协议下的原型**，也不能宣称 EPA 与原 CLIP 文本空间完全一致。带前缀原型作为敏感性/消融配置保留，文件 `/root/autodl-tmp/Text-space-Audit/run-03-v3-prefixed/encoder_mean_prototypes.npy`，SHA256 `5c9a93d246f32ed0259ccd2414a1743b46cdeca57c6fa52bbe3eb516b09231d7`。

当前 Val 只有 40 句，配对 CI 包含 0，选择证据有限。EPA 不应因为 Test 的 0.821 而被描述成“已证明最佳”；后续真实 EEG 对照与无情感损失消融仍是独立的研究问题。本轮仅重做冻结文本审计，**没有实现 EPA、没有读取 EEG、没有启动 GLIM 训练**。

服务器新结果位于 `/root/autodl-tmp/Text-space-Audit/run-03-v3-prefixed/` 和 `/root/autodl-tmp/Text-space-Audit/run-04-released-prefixed/`；根目录 `prefix_comparison.json` 保存配对 bootstrap、向量相似度和验证集选择记录。新增 SHA256 清单单独保存，旧清单与旧 run 不改动。
