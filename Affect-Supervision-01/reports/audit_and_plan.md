# Affect-Supervision-01：代码、数据与执行计划审计

日期：2026-11-11（北京时间）  
阶段：审计与单批预检已完成；未启动长训练。

## 1. 审计结论

**V3 checkpoint 与其原始 GLIM/V3 代码兼容，但与 EPA-01 的实验类及其初始化协议不兼容。** 用原 `GLIMMultitask.from_glim_checkpoint()` 严格加载 V3 epoch 46，215 个 state_dict 项全部匹配，missing/unexpected 均为 0。该 checkpoint 是 `A3-full`：已有离散情感头、128×256 球面 VQ 码本及由情感头预测产生的 Attribute Token；原训练权重为 `0.5 CLIP + 0.5 LM + 1.0 VQ + 0.2 discrete emotion CE`，原 commitment 权重为 0。生成路径在 ZuCo1 Task1 中使用 Attribute Token，Decoder 为冻结 Flan-T5-Large。

EPA-01 的 `EPAExperiment` 明确只接受 `A3-no-attribute`、离散情感 CE 权重 0，且从发布版 GLIM 加原始 VQ 初始化码本的 `common_init.pt` 起步。它不能原样加载本轮指定的 V3 epoch 46；改成 EPA-01 模型会移除 checkpoint 中的 `attribute.*` 参数并改变生成路径。因此不能以 EPA-01 现有 E0/E2、现有 smoke test 或现有 pilot 作为 B0/B1 的初始化和公平性证据。两组须在本目录以原 V3 `A3-full` 结构重新实现并重新做单批验证，模型共享权重严格从同一 V3 checkpoint 加载；EPA 投影是 B1 唯一新增模块。

第二个设计差异：EPA-01 的 DataModule 只对 ZuCo1 Task1 施加情感标签，但训练 loader 仍遍历 ZuCo1/ZuCo2 的全部任务及八种训练文本变体。本轮数据范围写明 ZuCo1 Task1-SR，故必须在本目录建立 Task1 专用过滤入口，让 B0/B1 只在同样的 3316 个 train EEG trial（各有八个原 GLIM 文本训练变体）上继续训练，并在 406/417 个 val/test trial 上评价。原 EPA-01 数据**划分可复用**，训练 loader 不可原样复用。

第三个配置差异：EPA-01 配置为 50 epochs，只有最佳/最新 checkpoint 保存，没有验证早停。本轮上限 20 epochs 且要验证早停，因此需要新增早停回调。既有 EPA 温度 `τ=0.07` 与损失权重 `λ=0.2` 可以作为本轮预设，不依据测试集调整。

**已确认方案 B：** 两组继续训练时均保留 V3 的离散情感 CE 权重 `0.2`、既有情感头与 Attribute Token；B1 仅额外加入权重 `0.2` 的 EPA CE。B0 是“不增加新的情感监督”的 V3 延续基线，**并非无情感监督的纯 GLIM**。最终结论只能归因于在既有 V3 情感监督基础上再加 EPA 的增量效果。

## 2. 来源、模型与环境

| 项目 | 核验结果 |
|---|---|
| 指定 checkpoint | `/root/autodl-tmp/A3-v2-V3-50-A800/run-02/checkpoints/epoch=46-step=93436.ckpt`，229,085,118 bytes，SHA256 与要求一致 |
| checkpoint 内容 | epoch 46，global step 93,436，`A3-full`，215 个状态张量；原 V3 代码严格加载通过 |
| EEG 侧 EPA 接口 | 原 Aligner 的 `q_x` 经 cross-attention 输出 `eeg_emb_vector`，维度 1024；B1 接 `Linear(1024,1024)`、L2 归一化、余弦相似度、温度缩放、CE |
| 生成路径 | V3 `A3-full` 在 EEG memory 前加由既有离散情感头预测产生的 Attribute Token；生成用 beam 2、最大 64 token，T5 Decoder 不改 |
| 文本模型 | 本地 Flan-T5-Large 权重 SHA256 已核对；`setup()` 调用 `requires_grad_(False)`，Encoder/Decoder 均冻结；原 CLIP 文本侧 `q_y` 保持原样 |
| GPU | NVIDIA A800 80GB PCIe，当前 0 MiB 使用、无 GPU 进程；`torch.cuda.is_bf16_supported()` 为 True |
| 软件 | PyTorch 2.3.0+cu121，CUDA runtime 12.1，Lightning 2.4.0；驱动 595.71.05 |
| 磁盘 | `/root/autodl-tmp` 50 GB；初审约余 17 GB，单批测试前约余 24 GB；正式运行前复查 |

## 3. 数据审计

原 EEG pickle 与审核标签通过 `data/revised_labels.py` 的 UID 加规范化文本逻辑重新匹配；生成的完整 manifest SHA256 与 E1 和 EPA-01 相同。Task1 过滤后逐条检查 EEG 形状、mask 非空和有效 mask 范围内的有限数值：**4139/4139 有效**，均为 `[1280,128]`，无无效 trial。400 个 UID 各有唯一规范化句子、唯一三分类标签及唯一 split；跨 split UID/规范化文本重叠为 0。

| Split | 独立句子 | EEG trial | Negative / Neutral / Positive（句子） | Negative / Neutral / Positive（trial） |
|---|---:|---:|---:|---:|
| Train | 320 | 3316 | 103 / 104 / 113 | 1094 / 1069 / 1153 |
| Val | 40 | 406 | 8 / 16 / 16 | 78 / 165 / 163 |
| Test | 40 | 417 | 12 / 17 / 11 | 121 / 181 / 115 |
| 合计 | 400 | 4139 | 123 / 137 / 140 | 1293 / 1415 / 1431 |

12 位被试均出现。按 Train/Val/Test 计数：ZAB 307/38/39、ZPH 305/38/39、ZKH 301/37/38、ZDM 302/35/37、ZMG 297/39/37、ZKB 296/34/37、ZJM 286/31/36、ZJS 272/35/30、ZKW 260/31/34、ZGW 245/29/34、ZDN 230/32/31、ZJN 215/27/25。完整核验见 `data_audit.json`。标签表示**文本情感倾向**，不能解释为被试真实情绪。

训练句子的每 UID trial 数为 2–12，中位数 11；EPA-01 对情感 CE 用训练 UID trial 数倒数加权，可在 B1 复用。固定 E1/EPA-01 split SHA256 为 `13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`。本目录另存 Task1 专用清单；它的字节 SHA 不应与含全部任务的原清单混淆。

## 4. 原型与 EPA-01 审计

EPA-01 使用冻结 Flan-T5-Large Encoder **最后层的无 `To English:` 前缀 masked mean**。先对每个训练句向量 L2 归一化，再按类取均值并 L2 归一化；类别顺序为 Negative、Neutral、Positive，张量 `[3,1024]`、`float32`、三个行向量单位范数。原型仅由 320 个 train UID（103/104/113）构建，Val/Test 未参与拟合；SHA256 为 `ceaaccee85eee260c8c6e540bf441fced6421b2912aa9014d246e0cd55f5f31d`。原型是独立辅助监督空间，与 GLIM 原 `q_y` CLIP 文本空间不同。EPA 分支只接 EEG 侧 `q_x` 表征；真实句子文本不进入 EEG 推理路径。

此前 Text-space Audit 已查看同一 40 句测试集的原型诊断结果，故这不是完全盲测的首见测试集。后续不得凭现有或新测试结果选择原型、温度、损失权重或 checkpoint。

## 5. 方案 B 结构和锁定配置

B0：`L_total = 0.5 L_CLIP + 0.5 L_LM + 1.0 L_VQ + 0.2 L_disc_CE`。B1：同一基础损失加 `0.2 L_EPA_CE`。两组均使用 V3 `A3-full` 生成路径，已有情感头与 Attribute Token 不移除、不重置。B1 在 EEG 侧 `eeg_emb_vector` 后接 `Linear(1024,1024)`，归一化并与冻结三原型计算余弦/`0.07` 的 CE；EPA 不读取推理时的真实句子文本。

训练配置锁定在 `configs/common.json`：seed `42/43/44`，Adam `lr=1e-4`、weight decay 0，batch Train/Val/Test `72/24/24`，BF16 mixed。用户于 2026-11-11 将本轮上限设为 **10 epochs**；按每 epoch `val/loss_lm` 早停（patience 4，min_delta 0），同指标保存 best 与 last。每组每 seed 均从**同一 V3 模型权重**重新初始化 Adam，不恢复 V3 optimizer。B1 投影随机初始化后重新设置训练随机种子；Task1 sampler 用配对 seed。完整配置及启动入口均在本目录。

## 6. 单批预检与公平性核验

在 A800 上读取同一训练 split 的真实 UID 2、5（均为 ZAB），B0/B1 的 215 个 V3 共享张量逐值相等；原型与指定 checkpoint 的 SHA256 在测试前后均一致。两组的 `L_CLIP=0.614746`、`L_LM=16.414280`、`L_VQ=0.000144`、`L_disc_CE=0.000529`、`L_GLIM=8.514513` 逐项完全相同；总损失 B0 `8.514763`、B1 `8.751634`，B1 `L_EPA=1.184353`。B1 的 EPA 投影为 `[2,1024]`，logits 为 `[2,3]`。

B0/B1 初始生成文本逐条相同；两组均保留长度 97 的 V3 Attribute Token memory。B1 EPA 梯度进入 EEG Encoder 和 EPA 投影，未进入文本侧 `q_y`；两组既有离散情感头、Attribute Token、VQ 码本均有非零梯度，冻结 T5 与原型无梯度。BF16 下损失有限，单批 CUDA 峰值约 3.83 GiB。原始 V3 checkpoint 与代码严格加载，真实 batch 前向/反向通过。完整结果见 `smoke_report.json`。

两组由同一 `Task1DataModule` 和同一 `common.json` 构建；运行时 Task1 manifest 与本地固定清单的 400 行 UID、phase、情感标签、trial 数和句子文本逐行一致。两组 optimizer 都实测为 Adam，学习率 `1e-4`、weight decay `0`。早停、batch、epoch 上限和生成设置由共同配置读取，训练 runner 对输出目录和预检报告做硬性校验。

本目录复制的 V3 `model/` 五个 Python 文件及 `data/` 两个 Python 文件与原本逐文件 SHA256 相同；所有实验改动写在独立的 `affect_model.py`、`task1_data.py` 和新 `scripts/`，未修改旧实验脚本。

## 7. 正式训练和评价计划

1. 正式训练前复查 GPU 进程、可用显存和磁盘，再按配对 seed 顺序运行 B0/B1；每组最多 10 epochs，验证 LM Loss 早停。用户现已授权启动本轮长训练。
2. 每 epoch 保存 CLIP、LM、VQ、离散情感、EPA（仅 B1）、总 loss、验证指标、训练时长和学习率；记录最佳 epoch、早停原因及 SHA。若两组实际早停 epoch 不同，报告实际步数。
3. 对各 seed 最佳 checkpoint 使用相同 beam 2/max 64 解码。保存 trial 级 UID、被试、参考文本和生成文本；计算 BLEU-1/2/4、ROUGE-L、BERTScore、teacher-forced LM Loss。情感报告 Macro-F1、Balanced Accuracy、各类 Precision/Recall/F1、混淆矩阵；共同的 V3 离散情感头作 B0/B1 同口径比较，B1 EPA 预测另报。
4. 测试集对同一被试内不同 UID 的 EEG 与 mask 作固定错配，保持目标文本和 prompt 不变；比较配对/错配 EEG 的生成指标和 LM Loss，逐被试报告。三个 seed 逐个列出，再汇总 mean、sample SD；按 UID 做配对差异和区间，重点看生成质量与 LM Loss 的权衡。测试集不用于模型选择或超参调整。

## 8. 当前停止点

已完成审计、建立本地与服务器独立目录、生成数据核验与 Task1 UID 清单；新目录内已有独立模型/数据代码、锁定配置、训练入口及通过的单批报告。六个顺序训练任务已于 **2026-11-11 20:52:03 北京时间**启动，目前仍在执行；尚无完整六组生成质量或统计结论。实时状态见服务器 `reports/training_status.json`。
