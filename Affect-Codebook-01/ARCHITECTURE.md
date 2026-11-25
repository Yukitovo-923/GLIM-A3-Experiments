# Affect-Codebook-01 架构与审计

V3 的情感条件与新专家共用 Decoder memory 的第一个位置。该位置之后的 96 个连续 EEG token，以及原始 memory mask，直接沿用 V3。三组 memory 的形状均为每 trial 97 × 1024。

| 分支 | 输入及运算 | 输出去向 |
|---|---|---|
| V3 连续语义路径 | EEG、有效帧 mask 和 V3 原提示经 EEG Encoder 得到 hidden。Aligner 产生连续 memory 和 1024 维句向量 | 96 个连续 token 进入冻结 Flan-T5；句向量同时供新分支使用 |
| V3 原属性路径 | 原 K=128 VQ 量化 hidden，原 disc_head 掩码池化并输出情感 logits | EmotionAttribute 对预测概率加权的三项 embedding 做投影和 sigmoid 门控，构成原 attribute token |
| A1 连续对照 | 标准化句向量，经线性投影和 L2 归一化；普通 MLP 读出三类 | 连续 256 维向量经过 fusion 加到原 token |
| A2 离散专家 | 同一标准化、投影和 L2 归一化；对 3 组共 12 个新码字计算相似度 | 全局最近码经过同规格 fusion 加到原 token |

GLIMMultitask.condition_memory 调用 emotion_outputs，随后用 attribute 对 disc_logits 的读出前置一个 token。该位置对有有效 EEG 帧的 ZuCo1 Task1 样本打开。真实标签在生成入口的参数表中没有位置。原 V3 的 detached probability 和已学习 gate 均沿用 checkpoint。A0 直接解码该 memory。A1/A2 只替换首位置的数值，残差投影初始化为全零。

V3 所有模块均为 eval 模式且冻结。冻结的 T5 仍传递对新增输入的梯度。训练阶段缓存原 EEG memory、mask 和句向量，避免重复更新或重算原模型。原始 EEG 推理入口见 mechanism.decode_eeg，输入为 EEG、mask 和原任务/数据集/被试提示。参考文本用于 LM 监督或离线评分。

**运行时核查。** 两个划分各自首批 24 条的纯 EEG memory、mask、句向量与 V3 的原生 shared_forward 完全相等。每批前 4 条的自由生成一致。训练完成后，三组又分别从验证集首批 24 个原始 EEG 重走生成路径，各有 24/24 个字符串与评分使用的缓存路径完全一致。

复现顺序为 prepare.py、pilot.py、mechanism.py、score.py，均在新实验目录使用 /root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python -Bu 执行。评分复用原项目的本地 BERTScore 包和权重。阶段权重为 stage1_heads.pt 与 stage2_heads.pt，包含两组新增模块和训练集标准化统计。V3 权重由 prepare.base_model() 严格恢复。
