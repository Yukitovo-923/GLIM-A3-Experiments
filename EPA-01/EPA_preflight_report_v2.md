# EPA-01 预检报告 v2

日期：2026-11-08（北京时间）。本轮完成独立 EPA 代码、E0/E1/E2 配置、公平性审计和真实 ZuCo 单批 smoke test；**未启动完整训练，也未更新任何原实验权重**。

## 1. 代码和实验对象

独立服务器目录：`/root/autodl-tmp/EPA-01/`。新增 `epa_model.py`、`epa_assets.py`、`epa_data.py`、`run_epa.py`、`smoke_epa.py`、`common.json`、`config_E0.json`、`config_E1.json`、`config_E2.json`。原 GLIM、V3 和 A0 代码与 checkpoint 未修改。三组共用 `EPAExperiment`、同一数据入口及同一原始生成路径。

原 GLIM 的 EEG Encoder 输出是 `[B,96,256]`。Aligner 的 `in_proj` 先生成 `[B,96,1024]`，EEG 侧可学习 `q_x` 经 cross-attention 得到 `eeg_emb_vector` `[B,1024]`；文本侧 `q_y` 另行得到文本查询向量。EPA 从 `eeg_emb_vector` 接一个独立 `Linear(1024,1024)` 和 L2 normalization，与三个冻结原型作余弦相似度并除以 `tau=0.07`。`q_y`、原 CLIP 和送入 T5 的 `eeg_embeds` 均未改动。三组统一使用 `A3-no-attribute`，不注入情感 Attribute Token。

原型：无 `To English:` 前缀的 Flan-T5-Large `encoder_mean`，来自 `/root/autodl-tmp/Text-space-Audit/run-01/encoder_mean_prototypes.npy`；类别顺序 negative、neutral、positive，形状 `[3,1024]`，三个向量均为单位范数，SHA256 `ceaaccee85eee260c8c6e540bf441fced6421b2912aa9014d246e0cd55f5f31d`。只由 Train 的 320 句计算；本轮仅复用和验证，未用 Val/Test 更新或选择原型。这是**独立的无前缀情感监督空间**，与 GLIM 原 CLIP 文本侧查询空间不同。

## 2. 共同初始化与数据审计

三组都严格加载 `/root/autodl-tmp/EPA-01/initialization/common_init.pt`，文件 SHA256 `b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e`，张量状态 SHA256 `ed0882f6c38db356a5ca30621f0af18362bd6ff5062dbf34566c89c00fc34c99`。该文件保留发布版 GLIM checkpoint 的 204 个匹配参数，以及**训练前** V3 spherical K-means 初始化资产的 3 个 VQ 参数（projection weight/bias、128×256 codebook）；原始 GLIM 参数与 V3 初始化完整文件逐张量相等。未使用 V3 epoch 46 情感训练后权重。

源资产 SHA256：发布版 GLIM `25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`；V3 原始 VQ 部分 `65c9be6c7f6f4b72cd5637a5381490cd927d8dd225a82b90824ef63dff7b9a39`；V3 完整初始化 `20529e073164c1999dbf3b5107a4181baef8de35b5c177326c44851b42cc5767`。共同初始状态含 214 个 state_dict 项，模型 trainable 参数 19,989,894；原型的 3,072 个值为冻结 buffer。E0/E1/E2 均严格还原同一状态张量，非情感参数与初始随机参数相同。

E1 原始 split manifest SHA256 `13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`，EPA 复现的 manifest SHA 完全一致。ZuCo1 Task1 为 400 个独立有标签句子、4,139 条 EEG trial；Train/Val/Test 分别 320/40/40 句、3,316/406/417 trial。句子级 negative/neutral/positive 数量分别为 Train `103/104/113`、Val `8/16/16`、Test `12/17/11`，总数 `123/137/140`。原审计通过 sentence UID 及规范化文本校验无跨 split 泄漏；Task2/3 不继承 Task1 情感标签。Train 每句 trial 数 2–12，中位数 11，12 位被试的 Train trial 数 215–307；情感 CE 使用 Train UID trial 数倒数加权。后续验证和测试同时保存 trial-level、按同句预测概率平均的 sentence-level、逐被试指标。E0 情感头从未受情感监督，其分类指标只能视作诊断，不是训练好的情感基线。

## 3. 损失和训练配置

共同非情感项：`L_base = 0.5 L_CLIP + 0.5 L_LM + 0 × L_original_commitment + 1.0 L_VQ`。VQ 使用同一 128×256 球面 K-means 初始码本、cosine quantization、`L_VQ = L_codebook + 0.25 L_vq_commitment`。无效位置按原 VQ mask 排除。

| 配置 | 总损失 | 情感监督入口 |
|---|---|---|
| E0 | `L_base` | 无 |
| E1 | `L_base + 0.2 L_disc_CE` | 原 V3 离散 VQ Emotion Head；按 UID 反比加权 |
| E2 | `L_base + 0.2 L_EPA_CE` | EEG 侧 Aligner `eeg_emb_vector` → EPA 线性投影 → 冻结文本原型；按 UID 反比加权 |

三组都用 seed 42、Adam、学习率 `1e-4`、weight decay 0、batch size Train/Val/Test `72/24/24`、BF16 mixed、50 epochs 的**预备配置**。训练入口每轮保存可恢复 `last.ckpt`，另保留 `val/loss_lm` 最佳 checkpoint；所有结果将放在独立输出目录。未启动上述 50 epochs。

## 4. 真实 ZuCo 单批 smoke test

结果：`/root/autodl-tmp/EPA-01/smoke-run-02/smoke_report.json`，状态 `PASS`，真实 Train Task1 的两条不同句子 UID `2`、`5`，batch size 2。E0/E1/E2 的基础 `L_CLIP=0.665039`、`L_LM=14.129774`、`L_VQ=0.000660`、`L_GLIM=7.397407` 在相同随机状态下逐项完全一致。三组总损失分别 `7.398067`、`7.616607`、`7.601229`，均有限。EPA 输入和投影 `[2,1024]`，EPA logits `[2,3]`；无效标签 `-100` 的 CE 和梯度 mask 检查通过。原 GLIM 前向结果逐项一致，原生成路径完成一次短文本生成，且没有 Attribute Token。

E2 `L_EPA` 对 EEG Encoder 和 EPA 投影的梯度均非零，独立梯度检查确认不经过文本侧 `q_y`；T5 和原型没有梯度。总损失反向传播时 E0/E1/E2 的 EEG Encoder 梯度范数分别约 `5664.529/5664.524/5664.415`；E1 离散情感头梯度范数 `0.0914`，E2 EPA 投影梯度范数 `0.2700`；三组 VQ codebook 梯度均约 `0.000227`。E0/E1 的 EPA 投影梯度为零。源 checkpoint、VQ 初始化与原型 SHA 在测试前后相同。单批初始 codebook perplexity `80.73`，不能据此断言长期无坍缩。

首次测试目录 `smoke-run-01` 因测试代码把 Encoder 的冻结参数也传给 `torch.autograd.grad` 而中断；改为只检查可训练参数后，完整 `smoke-run-02` 通过。该修复未改模型前向或原始训练代码。真实 smoke 的 CUDA 峰值为 **2.20 GiB（batch 2）**；测试进程退出后 A800 显存占用 0 MiB。

## 5. 公平性边界、资源与风险

共同初始化、split、训练预算、基础优化器、batch、非情感损失、VQ 初始状态和生成器输入均一致。**E1 与 E2 仍不能隔离“文本原型目标”本身的因果效应**：E1 的情感梯度进入 VQ 离散头，E2 的情感梯度由 EEG 侧 Aligner 输出直接回传，监督表征层级和梯度路径也不同。三组可比较整体方案，若以后要单独归因于 prototype target，还需同一输入层级的 CE 对照。E0 情感头不训练。原始发布版 GLIM 的预训练数据来源是否曾覆盖这 400 句，现有文件未充分证明；后续结果应按此限制解释。

当前服务器 A800 80 GB；单批 B2 峰值 2.20 GiB，**尚未测量 B72 正式训练峰值，不能按 B2 线性外推**。同配置 V3 曾在 A800 运行，但 EPA 要在首次正式运行时再次监测显存。数据盘 `/root/autodl-tmp` 共 50 GB，当前剩余约 20 GB；入口采用 best+last checkpoint 策略以避免三组保存全部 50 轮权重占满磁盘。实际每轮耗时和完整 checkpoint 占用尚未测量。训练前还应检查输出目录唯一、可用磁盘、显存和已有任务。

## 6. 后续启动命令（本轮未执行）

三组应在单张 A800 上**顺序**运行；以下命令均使用相同 `common_init.pt` 与 smoke 证明，输出目录独立。按需先执行 E0，再执行 E1、E2：

```bash
export PYTHONPATH=/root/autodl-tmp/GLIM-main-A3-v2:/root/autodl-tmp/EPA-01
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
EPA=/root/autodl-tmp/EPA-01
$PY $EPA/run_epa.py fit --arm-config $EPA/config_E0.json --common-init $EPA/initialization/common_init.pt --smoke-report $EPA/smoke-run-02/smoke_report.json --output $EPA/runs/E0-01
$PY $EPA/run_epa.py fit --arm-config $EPA/config_E1.json --common-init $EPA/initialization/common_init.pt --smoke-report $EPA/smoke-run-02/smoke_report.json --output $EPA/runs/E1-01
$PY $EPA/run_epa.py fit --arm-config $EPA/config_E2.json --common-init $EPA/initialization/common_init.pt --smoke-report $EPA/smoke-run-02/smoke_report.json --output $EPA/runs/E2-01
```

本轮没有创建上述 `runs` 目录，也没有启动训练。三个配置的实际性能仍未知；当前证据仅支持工程预检通过。

## 7. 可核验文件

- `/root/autodl-tmp/EPA-01/initialization/initialization_audit.json`：资产与共同初始状态审计。
- `/root/autodl-tmp/EPA-01/smoke-run-02/data_audit.json`：句子、trial、被试分布。
- `/root/autodl-tmp/EPA-01/smoke-run-02/split_manifest.csv`：与 E1 完全相同的划分。
- `/root/autodl-tmp/EPA-01/smoke-run-02/smoke_report.json`：三个分支真实单批结果。
- `/root/autodl-tmp/EPA-01/EPA_preflight_report_v2.md`：本报告。

代码与配置文件本地和服务器 SHA256 逐文件一致；服务器 Python 编译检查和训练入口 `--help` 通过。完整训练与完整验证/测试尚未执行。
