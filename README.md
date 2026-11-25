# GLIM A3 Experiments

基于 [GLIM](https://github.com/justin-xzliu/GLIM) 的 EEG-to-Text 实验代码，整理自 A3 实例，供代码阅读、方法讨论和实验审查。

本仓库保留 GLIM 基座、VQ 变体、EPA 情感监督、情感码本及线性探针等实现，以及部分实验总结。源码快照日期为 **2026-11-25（北京时间）**。各实验使用过的代码副本保留在原目录中，方便对照配置与实现。

## 从哪里看起

| 想了解的内容 | 推荐入口 |
| --- | --- |
| EEG 编码器、对齐与语言生成 | [GLIM 主模型](GLIM-main-A3-v2/model/glim.py) |
| 连续/离散情感分支与实验变体 | [多任务模型](GLIM-main-A3-v2/model/glim_multitask.py)、[情感头](GLIM-main-A3-v2/model/emotion_head.py) |
| VQ 实现与初始化 | [TokenVQ](GLIM-main-A3-v2/model/vq.py)、[初始化](GLIM-main-A3-v2/initialize_vq_v2.py) |
| 数据读取、句子划分与标签核验 | [数据模块](GLIM-main-A3-v2/data/datamodule.py)、[标签审计](GLIM-main-A3-v2/data/revised_labels.py) |
| GLIM 与 EPA 的训练对照 | [Affect-Supervision-01](Affect-Supervision-01/README.md) |
| 情感码本架构与实验结论 | [架构](Affect-Codebook-01/ARCHITECTURE.md)、[总结](Affect-Codebook-01/SUMMARY.md) |
| 监督对比学习 Probe | [SupCon 总结](SupCon-01/SUMMARY.md)、[实现](SupCon-01/probe.py) |
| 句子级可读性与混淆因素 | [E1 报告](E1-V3-Long-Readability/E1_report.md)、[E1.1 报告](E1.1-V3-Readability/E1.1_report.md) |
| 文本空间与输入模板诊断 | [文本空间审计](Text-space-Audit/REPORT.md)、[模板比较](Text-space-Audit/PREFIX_COMPARISON_REPORT.md) |

完整目录与实验依赖见 [实验导航](docs/EXPERIMENTS.md)，环境、外部资源与运行示例见 [运行说明](docs/SETUP.md)。

## 内容与验证范围

- 包含 Python/Shell 源码、实验配置、依赖清单、许可证和部分 Markdown 报告。
- 数据集、标签表、划分清单、预训练模型、checkpoint、特征缓存、逐条预测与训练日志需要另行准备。本仓库不包含这些资源。
- 历史脚本保留 A3 的绝对路径和资源 SHA256 校验。迁移时需要逐项配置资源路径；克隆仓库本身不足以复现实验。
- 报告中的数字来自已有实验记录；本次整理只检查源码与配置，不重新训练模型。各报告的验证集、测试集及种子范围以报告正文为准。

整理时新增导航、说明和仓库维护文件，重写情感监督目录的入口说明，并修复 `initialize_vq_v2.py` 首行多余空格引起的语法错误；模型和训练逻辑保持原样。原实例中的训练目录和文件未被移动或修改。[来源清单](docs/SOURCE_MANIFEST.json)记录导出文件的 SHA256 与整理时的差异。

## 上游来源与许可

本项目基于 Xiaozhao Liu、Dinggang Shen、Xihui Liu 的 GLIM 实现：

> Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation (2025).

上游仓库：[justin-xzliu/GLIM](https://github.com/justin-xzliu/GLIM)；论文：[arXiv:2505.17099](https://arxiv.org/abs/2505.17099)。

沿用上游 **CC BY-NC-SA 4.0**，参见 [LICENSE.txt](LICENSE.txt) 和 [许可原文](https://creativecommons.org/licenses/by-nc-sa/4.0/)。使用、修改或分享时应保留署名，限非商业用途，并以相同条款分享改编内容。数据和模型资源的使用条件由各自提供方规定。
