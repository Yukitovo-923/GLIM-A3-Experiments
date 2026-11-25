# 实验导航

目录名与 A3 的源码目录对应。保留各实验自己的模型副本，是因为它们记录不同阶段的实现和配置，阅读时应同时核对所属目录。

| 目录 | 作用 | 入口 |
| --- | --- | --- |
| `GLIM-main-A3-Revised/` | A3 Revised 的连续/离散多任务基线 | `run_multitask.py`、`model/glim_multitask.py` |
| `GLIM-main-A3-v2/` | VQ 距离、初始化和 Pilot 变体，后续 V3 的代码基座 | `initialize_vq_v2.py`、`vq_pilot.py`、`model/vq.py` |
| `A3-v2-V3-50-A800/` | V3 长训练恢复与评估代码；保留嵌套的评估脚本 | `resume_v3_50_a800.py` |
| `A3-v2-V3-50-A800-stage-reports/` | 训练阶段诊断工具 | `stage_monitor.py` |
| `E1-V3-Long-Readability/` | 冻结表征上的句子级线性探针、噪声和混淆控制 | `E1_report.md`、`e1_linear_readability_v2.py` |
| `E1.1-V3-Readability/` | 原情感头复测与 Pooling 消融 | `E1.1_report.md`、`e11_diagnostic.py` |
| `E1.2-A0-V3-Sentiment-Supervision/` | A0/V3 对照的资源审计；报告记录该比较未运行 | `E1.2_report.md` |
| `Text-space-Audit/` | 文本空间、情感原型及输入模板分析 | `REPORT.md`、`text_space_audit.py` |
| `EPA-01/` | EEG 侧情感原型对齐及早期训练对照 | `epa_model.py`、`run_epa.py` |
| `EPA-01-Continuation-50/` | EPA 后续训练队列与结果汇总脚本 | `queue50.sh`、`summarize_E2_readonly.py` |
| `Affect-Supervision-01/` | B0/B1 情感监督对照及后续恢复、学习率、生成目标诊断 | `README.md`、`models/affect_model.py` |
| `Affect-Codebook-01/` | 冻结 V3 上的分组情感码本与生成干预 | `SUMMARY.md`、`prepare.py`、`pilot.py` |
| `SupCon-01/` | 在已有冻结特征上比较 CE 与 CE + SupCon | `SUMMARY.md`、`probe.py` |

## 阅读顺序

1. 先阅读 `GLIM-main-A3-v2/model/glim.py`，理解 EEG 编码器 → 对齐层 → T5 的路径。
2. 阅读 `model/glim_multitask.py`、`model/vq.py` 和 `model/emotion_head.py`，理解离散表征与情感监督的接入方式。
3. 阅读 `Affect-Supervision-01/models/affect_model.py`，对照 `configs/common.json` 核对损失和资源约束。
4. 根据兴趣进入码本、SupCon、线性探针或文本空间诊断，并结合各目录报告理解结果的适用范围。

## 实验之间的资源依赖

- V3、EPA 及后续情感实验通常依赖 A3 Revised 的 Flan-T5 和配对 EEG 文件。
- 情感监督对照依赖 V3 epoch 46 checkpoint、E1 的句子划分清单，以及文本空间审计得到的原型。
- `Affect-Codebook-01/prepare.py` 读取情感监督的诊断配置并生成冻结特征；`SupCon-01/probe.py` 再读取这些缓存。
- `Affect-Supervision-01/` 下 `*_run01`、`*_probe01` 等子目录是实验代码快照，不是可随意互换的公共库。

部分历史报告链接到未随源码分享的预测、图表或运行产物；这些链接仅表示原实验的产物位置。
