# SupCon-01 结论

**在本次冻结 V3 特征的轻量 Probe 中，SupCon 没有稳定提高未见句子的情感恢复，也没有缓解 Positive 偏置。建议按停止标准暂缓接入生成或扩大情感专家。**

| 问题 | 本次证据 |
| --- | --- |
| 跨 UID 泛化 | 三种子平均验证 Macro-F1：CE 0.2145，CE＋SupCon 0.2162；Balanced Accuracy：0.3264、0.3333。 |
| Positive 偏置 | 种子 42、43 均保持 36/40 个 Positive。种子 44 从 37/40 增到 39/40，Negative 和 Neutral 各无一个判对。 |
| 跨句情感结构 | 训练句之间的同/异类分离略增。未见验证句之间的同−异余弦差均低于 0.006。六组验证→训练最近邻一致率均为 40%，等同多数类基线。 |
| 三个种子 | 42、43 的两项验证分类指标各自不变；44 的微弱提升只伴随 Positive recall 增加。近邻命中数三个种子均无提升。 |
| 进入生成阶段 | 依据本轮验证证据暂缓。当前比较检验冻结 V3 特征、此投影、固定 λ/τ 与预算的组合；未测生成质量或其他对比任务。 |

三组配对均使用相同初始参数和 batch，每组 52 步。CPU 完成数据审计、单批 smoke 及全部 6 组训练。复用 3,316 个训练 trial（320 UID）与 406 个验证 trial（40 UID），未运行 V3 或 Decoder，也未评估 Test。训练数据的高可分性和验证失败之间的成因仍未识别。

详情见 [CONFIG.md](CONFIG.md)、[BATCH_SAMPLING_AUDIT.md](BATCH_SAMPLING_AUDIT.md)、[RESULTS.md](RESULTS.md) 与 [EMBEDDING_ANALYSIS.md](EMBEDDING_ANALYSIS.md)。
