# 下一轮设计前六项检查（进行中）

检查日期：2026-11-13。只读审计旧实验；新诊断代码仅写入 `Affect-Supervision-01/scripts/`。不使用现有测试集选择新超参数。

## 1. 优化器与早停 — 已核验

- 六组最优验证 LM Loss 均在完成第 1 个 epoch（checkpoint 文件的 `epoch=0`）；完成第 5 个 epoch 后早停。
- epoch 0 → 4 的验证 LM Loss 增量：B0 seed 42/43/44 为 `+0.823/+0.839/+0.404`；B1 为 `+0.735/+1.012/+0.879`。因此退化是两组共同现象，不能单独归因于 EPA。
- 源 V3 epoch 46 checkpoint 包含 214 项 Adam 参数状态，参数组 LR `1e-4`、weight decay 0；本轮 `run_train.py` 只调用 `read_v3()` 加载模型权重并新建 Trainer，不恢复该 optimizer state。本轮 LR 仍为 `1e-4`，与源 V3 参数一致，但对已经训练 93,436 step 的权重相当于重新开始 Adam 的动量历史。
- 源 V3 与本轮 UID split manifest 的 SHA256 完全相同：`13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`，未发现源 checkpoint 将本轮 test UID 作为 train UID 的跨 split 泄漏。源 V3 训练包括 ZuCo1 task1/2/3、ZuCo2 task2/3；本轮只继续训练 ZuCo1 task1。训练每 EEG trial 有 8 种目标文本变体，验证使用原句，目标分布也不同。
- `training_curve.csv` 中 train loss 来自 Lightning 在 `training_step` 的默认日志，而 val loss 来自 `validation_step`；二者不能直接按同口径比较为泛化间隙。需要在同一 `eval()` 口径下对训练原句与验证原句重新计算 LM Loss，才能区分 optimizer 重置与目标分布的影响。

**结论：**续训方式和数据范围变化是明确的候选原因，当前证据不能量化各自贡献。下一轮只用新的验证划分比较较小 LR、恢复 optimizer 状态与目标文本策略。

## 2. EPA 几何与类别塌缩 — 部分完成，运行时检查待 GPU

- 冻结训练集原型的 Negative/Neutral/Positive 两两余弦：Neg–Neu `0.9750`、Neg–Pos `0.9211`、Neu–Pos `0.9562`。三向量共享很大的方向分量；这提示类别分离可能困难，但单凭原型余弦不能证明必然塌缩。
- 句级标签数量（Negative/Neutral/Positive）：train `103/104/113`、val `8/16/16`、test `12/17/11`。验证集中 Negative 只有 8 句，按类指标不稳定；训练集略偏 Positive，但不足以单独解释测试预测的大幅 Positive 倾向。
- B1 三个种子最佳 epoch 的验证 EPA CE 分别为 `2.3343/2.3523/2.3374`，高于三类均匀随机预测的 CE `ln(3)≈1.099`。现有测试预测仅作既有结果说明：EPA 的平均 Positive 概率为约 `0.655/0.732/0.704`，而测试 trial 的真实 Positive 占比约 `0.276`，类别读出明显偏斜。
- 已准备 `scripts/audit_epa_geometry.py`，只读取训练/验证 EEG，输出每类 logit、预测比例、投影前后到原型的余弦和 UID 级 Macro-F1。梯度规模/方向尚未实测。

## 3. 原 V3 情感监督上限 — 结构结论已核验

- B0/B1 均从 `A3-full` 初始化，保留原离散情感 CE 权重 `0.2`、VQ、情感头与 Attribute Token；B1 只增加 EPA CE `0.2`。所以本轮无法估计“没有任何情感监督”的效果。
- 在已有 V3 checkpoint 上把 CE 权重改为 0，仍保留了预训练得到的情感参数和 Attribute Token，因此也不是干净的无情感对照。若要回答该问题，需要共同的更早初始化点和严格相同的生成路径，另立实验。

## 4. EEG 对文本的可识别作用 — 既有结果已复核，多次验证错配待 GPU

- 既有测试中，正确 EEG 相比同被试不同 UID 错配 EEG 的 LM Loss 平均下降 B0 `0.32718`、B1 `0.35345`；六个 arm×seed 组合方向一致。
- 换 EEG 后实际生成字符串几乎总改变：B0 三种子有 `1/4/2` 条、B1 有 `1/0/2` 条（每组 417 条）保持完全相同；然而 BERTScore 与 ROUGE-L 没有稳定正确配对优势。故 EEG 影响解码，但未证明恢复了对应句子的语义。
- 既有 40 UID bootstrap 的 BERTScore 正确减错配区间跨 0。下一步在验证集进行多次独立同被试错配；现有测试集不再用于设计选择。

## 5. 推理文本依赖 — 静态审计完成，运行时反事实待可用环境

- `GLIM.shared_forward()` 在推理评估中仍编码 `input text`，用于 CLIP/commitment 计算；`Aligner.forward()` 仅在 `self.training and use_y_mask` 时把文本 mask 施加到 EEG。eval 模式下 `eeg_embeds` 由 EEG Encoder 和 EEG 侧对齐模块产生；V3 `condition_memory()` 的生成 mask 也不取真实文本 mask；Decoder 的生成输入是 EEG memory。
- 已准备 `scripts/audit_text_dependence_cpu.py`：固定验证 EEG、mask、prompt、target，仅交换参考 `input text`，逐值比较 EEG memory、memory mask、LM Loss 和生成 token。当前 SSH 容器无 GPU、cgroup 内存仅 2 GiB，加载约 3 GiB 的 Flan-T5 权重时进程被终止，没有运行时结论。

## 6. 封存测试集 — 来源审计已核验，需新数据或新训练协议

- 源 V3 的 train/val/test UID split 与本轮完全一致，未发现源 checkpoint 的 train/test UID 交叉。B0/B1 本轮也只按 val LM Loss 选 checkpoint。
- 先前 Text-space Audit 已查看同一 40 个 test UID 的原型诊断，因此本轮测试结果并非完全首见盲测。当前 400 个 Task1 UID 均已在源 V3 的 train/val/test 流程中使用；仅从这 400 UID 内重新抽取“新测试集”不能对当前 checkpoint 构造真正未触及的封存测试。需要外部新句子/新数据，或从更早的未接触这些 UID 的共同初始化点重新设计训练与封存划分。

## 环境阻塞

2026-11-13 当前 SSH 入口可连接但 `torch.cuda.is_available()==False`、`nvidia-smi -L` 无设备；cgroup 内存上限 `2,147,483,648` bytes。训练/验证表征、梯度和文本反事实运行时检查等待 A800 环境恢复。不会把未完成项写成已通过。
