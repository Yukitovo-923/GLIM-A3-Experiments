# EPA-01 预检报告：暂停实现

**日期：2026-11-07（北京时间）**  
**状态：BLOCKED — 未实现 EPA 模块，未加载真实 EEG batch，未运行 smoke test，未启动训练。**

## 阻塞原因：现有文本原型与 GLIM 训练文本输入协议不一致

Text-space Audit `run-01` 的 `text_space_audit.py` 第 113、117 行直接将 manifest 中的原始 `input text` 交给 Flan-T5 tokenizer。已保存的句子 CSV 中，例如 UID 0 的输入为 `Presents a good case ...`。但 GLIM 的 `data/datamodule.py` 第 296–298 行，在每条原始句子前拼接 `To English: <MASK>`，使模型实际接收 `To English: Presents a good case ...`；`model/glim.py` 第 181、234 行再将这份带前缀的 `batch['input text']` 送入 Flan-T5 Encoder 和 Aligner。

因此，`run-01/encoder_mean_prototypes.npy` **文件完整、标签/UID/划分正确，但其数值来自无前缀的原始句子**，不是 GLIM 训练时文本 Encoder 输入分布下的 masked-mean 原型。`run-01/clip_text` 也不是实际 GLIM CLIP 对齐文本输入的向量。先前 Text-space Audit 报告把后者描述成 GLIM 实际主对齐空间，此处明确更正：它是**用 GLIM Aligner 编码的无前缀句子空间**。原先的分类分数仍是该无前缀文本空间的实测结果，但不能直接解释为训练时 `To English:` 输入下的结果。

用户要求“发现结构、初始化或数据问题立即停止”。这属于 EPA 目标空间定义问题；若继续实现或为了通过 smoke test 直接接入缓存原型，将把不一致的文本输入协议固化进实验。因此本轮在只读审计后暂停。没有修改 GLIM/V3/A0 代码、checkpoint、原型文件或历史结果。

## 已完成的只读结构审计

1. **原始 GLIM EEG 表征。** EEG Encoder 输出 `H:[B,96,256]`；`model/modules.py` 的 Aligner 先以 `in_proj` 将其变为 `[B,96,1024]`，再由 EEG query `q_x` 的 cross-attention 得到 `eeg_emb_vector:[B,1024]`。双向 CLIP 损失直接对这个句级 EEG 向量和文本 query `q_y` 的句级向量做 L2 归一化、余弦相似度与交叉熵。若采用“在原 CLIP 句级空间上增加独立线性 EPA 投影”的方案，合理输入是 `eeg_emb_vector`，EPA 线性层应为 `1024→1024`；H 本身不是原 CLIP 损失使用的 EEG 向量。
2. **CLIP/LM 位置。** CLIP 在 `model/modules.py` 的 `Aligner.align_emb_vector()`；LM 在 `model/glim.py` 的 `text_decoder_forward()`。原 GLIM 组合在 `model/glim.py` 的训练步骤中为 `0.5·CLIP + 0.5·LM + 0·原 commitment`。EPA 应只作为额外辅助损失，不能替代这些路径。
3. **V3 情感与 VQ。** `model/glim_multitask.py` 的 `A3-full` 对 96 个有效 query 做 VQ、masked mean pooling、离散 EmotionHead，并以 Task1 有标签样本的交叉熵参与总损失；配置中离散情感权重 0.2、VQ 权重 1、原 commitment 权重 0。`model/vq.py` 使用 256→256 projection、128×256 可学习码本、cosine 最近邻、STE；码本损失更新码本，STE 将离散情感梯度送回 projection 与共享 EEG Encoder。`A3-full` 还把情感预测产生的属性 token 注入生成器，E0/E1/E2 若不统一禁用此路径，会出现生成输入结构混杂。
4. **数据关联。** `ZuCoDataset.__fetch_from_df()` 保留 `text uid`、`subject` prompt、EEG trial、mask、情感标签；训练时 8 个改写 target 通过 `collate_fn` 展开，GLIMSampler 保证**同一 batch** 的 UID 不重复，但不能单凭这一点证明**整个 epoch** 每个句子权重相等。同句不同被试 trial 数和 subject 不平衡尚未重新审计；本轮未读取 14GB EEG pickle。现有 E1 manifest 已核实 Task1 为 320/40/40 个句子，无 UID/规范化文本跨划分重叠。
5. **可用初始化。** 当前实例未找到独立 A0 训练 checkpoint。发布版 GLIM `/root/autodl-tmp/GLIM-main-A3-Revised/checkpoints/glim-zuco-epoch=199-step=49600.ckpt` 可读，SHA256 `25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`；V3 epoch 46 受过情感监督，不能作为三组共同起点。V3 spherical K-means 初始化文件 `/root/autodl-tmp/GLIM-main-A3-v2/runs/A3-v2-initialization/V3/initialization.ckpt` SHA256 `20529e073164c1999dbf3b5107a4181baef8de35b5c177326c44851b42cc5767`。它的 EEG Encoder、Aligner、PromptEmbedder 共 **204 个基座参数键**与发布版 GLIM **逐张量完全相等**；另含 VQ projection 和 codebook，但也包含随机情感头、属性模块。独立的 VQ projection+codebook 文件 SHA256 `65c9be6c7f6f4b72cd5637a5381490cd927d8dd225a82b90824ef63dff7b9a39`。发布版 GLIM 加这组 VQ 初始权重有条件地可作为三组共同初始化，仍需工程实现及保存完整初始化审计。
6. **缓存原型本身。** `Text-space-Audit/run-01/encoder_mean_prototypes.npy` SHA256 `ceaaccee85eee260c8c6e540bf441fced6421b2912aa9014d246e0cd55f5f31d`，来自 320 个 train 句、1024 维、三类；文件校验与标签来源均通过。**问题在文本输入前缀，而非损坏、标签不符或 Val/Test 泄漏。**

## 公平对照的条件方案（尚未实施）

E0/E1/E2 可以在同一份发布版 GLIM 基座与同一份 V3 VQ projection+codebook 上构建共同初始状态。三组均保留 VQ 和相同 VQ/CLIP/LM 权重；统一使用**不注入情感属性 token**的生成路径，以避免 E1 的情感头改变生成器输入。E0 关闭情感损失；E1 使用原 V3 离散 EmotionHead 的交叉熵；E2 使用额外线性 EPA 投影及冻结文本原型的交叉熵。三组使用相同 UID 划分、句子均衡规则、随机种子、batch size、Adam、学习率和训练步数。原始 GLIM 主生成代码保持不变。此设计仍需要先确定原型文本输入协议，并完成真实 EEG trial/subject 权重审计。

## 10 项问题的直接回答

| 问题 | 当前结论 |
|---|---|
| 1. GLIM 原 EEG 表征是哪一个？ | CLIP 使用 Aligner 的 `q_x` 汇聚向量 `[B,1024]`，由 Encoder `H:[B,96,256]` 产生。 |
| 2. 为什么选它作 EPA 输入？ | 它是原 CLIP 使用的句级 EEG 语义向量，单独 `Linear(1024,1024)` 可增加辅助约束而不替换 CLIP。方案未实现。 |
| 3. `encoder_mean` 原型是否成功复用？ | SHA/UID/标签核验通过；**未接入 EPA**，因无前缀文本与 GLIM 带前缀文本输入不一致。 |
| 4. A0 是否适合共同初始化？ | 独立 A0 checkpoint 当前不可访问，无法判定；发布版 GLIM 与 V3 VQ 初始化有条件可用。 |
| 5. EPA 接在何处？ | 拟接在 `eeg_emb_vector:[B,1024]` 后，独立线性投影；未写代码。 |
| 6. 三组能否公平？ | 在统一无 attribute token 路径、共同初始化和同一数据权重协议下**原则上可以**；尚未构造和验证。 |
| 7. 真实数据 smoke test？ | **未运行**，预检阻塞发生在数据加载前。 |
| 8. 梯度？ | **未测**；代码审计表明 VQ STE 理论上允许梯度回到 Encoder，但 EPA 梯度未实测。 |
| 9. 当前风险？ | 原型文本输入协议不一致；独立 A0 权重缺失；subject/trial 权重未审计；发布版 GLIM 预训练 UID 溯源缺失。 |
| 10. 下一步训练命令？ | **尚无有效命令**。需先解决原型输入协议、实现并通过真实单批 smoke，再生成三个可执行的独立训练命令。 |

## 恢复条件

优先在同一 400 句、原始 UID 划分上，用**实际 GLIM `To English:` 输入文本**重算一次冻结 Flan-T5 `encoder_mean` 原型，并与现有无前缀原型做相似度和验证/测试分类对照。若研究设计明确要把无前缀文本语义空间当作**外部辅助目标**，则应单独命名该条件，不能继续称为“与 GLIM 训练文本输入一致的原型”。确定条件后再实施 EPA 模块、句子/被试权重和真实 batch smoke test。不得覆盖旧 Text-space Audit 结果。
