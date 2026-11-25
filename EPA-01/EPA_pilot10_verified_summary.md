# EPA-01 三组 10-epoch Pilot 核验摘要

核验时间：2026-11-08 21:50（北京时间）。E0、E1、E2 均已完成 epoch 0–9；三个 `last.ckpt` 内部均记录 **epoch 9、global step 19,880**，含模型参数、Adam optimizer 和 Lightning loop 状态。自动 W&B 曲线/原汇总显示 19,879，是日志在最后一步更新前采样所致；**恢复位置以 checkpoint 的 19,880 为准**。后台训练与监控进程均已退出，GPU 占用 0 MiB，无 incident 文件。

| 实验 | 验证 LM Loss（epoch 9） | 句子级 Macro-F1 | Balanced Accuracy | 有效码 | Perplexity | 最终 checkpoint SHA256 |
|---|---:|---:|---:|---:|---:|---|
| E0 | 7.4624 | 0.1111* | 0.3333* | 124/128 | 97.11 | `36ecf0b6807c8cc6155c55ab3833796ea58ba3df751b7ab3f9ab4410dddad2b9` |
| E1 | 7.3656 | 0.3221 | 0.3333 | 123/128 | 73.30 | `356d980e562b1642771b0fed588fd231ebcc1c94ca3105708e2d17d6674ac080` |
| E2 | 6.9900 | 0.2348 | 0.3125 | 124/128 | 97.27 | `04b4cf703a5bbbe6ac28fcc206c43e3e8a8773ca67fba0a3678ab0ab1d4a682e` |

\* E0 情感头未受情感监督，分类结果只是未训练头的诊断数值，不能作为有意义的情感基线。E1、E2 的验证情感结果都偏弱，10 epochs 不足以证明 EPA 改善情感表征。E1 的监督来自 VQ 离散头，E2 来自 Aligner EEG 侧原型投影，梯度作用层级不同；不能将两者差异单独归因于文本原型。

最终 checkpoint：

- `/root/autodl-tmp/EPA-01/runs/E0-10-02/checkpoints/last.ckpt`
- `/root/autodl-tmp/EPA-01/runs/E1-10-02/checkpoints/last.ckpt`
- `/root/autodl-tmp/EPA-01/runs/E2-10-02/checkpoints/last.ckpt`

各组独立 `training_report.json`、`training_curve.csv`、逐轮 `metrics/val_epoch=*.json` 及 `best` checkpoint 保存在对应 `runs/E*-10-02/`。自动生成的原汇总 `/root/autodl-tmp/EPA-01/EPA_pilot10_comparison_r2.md` 保留未覆盖。本摘要只修正 checkpoint global step 的解读，不改训练指标或历史实验文件。
