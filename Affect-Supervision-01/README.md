# Affect-Supervision-01

从 V3 epoch 46 初始化，对照 B0（继续 GLIM）与 B1（继续 GLIM 并加入 EEG 侧 EPA），研究情感监督对 EEG-to-Text 的增量影响。

## 推荐阅读

- [实验模型](models/affect_model.py)、[数据模块](models/task1_data.py)
- [共同配置](configs/common.json)、[B0](configs/B0_Baseline.json)、[B1](configs/B1_EPA.json)
- [训练入口](scripts/run_train.py)、[单批核验](scripts/smoke_b0_b1.py)
- [测试推理](evaluation/infer.py)、[评分](evaluation/score_predictions.py)、[汇总](evaluation/aggregate_results.py)
- [初期对照结果](reports/final_comparison.md)、[审计和计划](reports/audit_and_plan.md)

## 后续诊断代码快照

| 目录 | 用途 |
| --- | --- |
| `step_level_diagnosis_run01/` | 逐步训练及梯度诊断 |
| `optimizer_resume_probe01/` | 优化器恢复与 Lightning 对照 |
| `optimizer_resume_run01/` | 恢复训练状态的独立实验 |
| `lr_1e5_run01/` | 学习率调整实验 |
| `canonical_target_run01/` | canonical target 与后续验证分析 |

这些子目录保留各自源码和配置，阅读时应核对实际选用的模型副本。仓库中的初期对照报告不代表后续所有分支的最终状态。

## 资源和运行

配置保留原服务器路径、checkpoint SHA256 和数据划分核验。运行需要另外准备 EEG/标签 dataframe、V3 epoch 46 checkpoint、Flan-T5、E1 split manifest 和文本情感原型。预测、缓存、权重、逐 epoch 日志及逐条数据不包含在分享仓库中。部分历史报告会引用这些外部产物。

环境及迁移注意事项见[仓库运行说明](../docs/SETUP.md)。
