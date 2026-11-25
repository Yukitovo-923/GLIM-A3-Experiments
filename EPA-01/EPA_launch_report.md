# EPA-01 正式训练启动核验（2026-11-08 14:18 北京时间）

状态：**E0 正在训练；E1、E2 已排入同一个后台顺序脚本，尚未启动。** 无完整三组结果，不能提前汇总最终指标。

- A800 80 GB；E0 batch 72、BF16 实际运行时显存约 23,133 MiB，GPU 利用率采样 92%。没有另行运行 batch 72 显存预测实验。
- 数据盘启动前可用 20,609,036,288 bytes；启动后、首轮 checkpoint 保存后的采样约 20,121,907,200 bytes。监控低于 5 GiB 时停止序列。
- `common_init.pt` SHA256 `b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e`；E0/E1/E2 同一绝对路径，不使用 V3 训练后 checkpoint。
- 后台顺序脚本 PID **3675**，E0 训练 PID **3682**，独立监控 PID **5625**；均脱离 SSH 运行。
- 最新监控采样：epoch **1**、global step **2449**；离线记录中的早期 LM、CLIP、VQ、总 loss 均有限，未触发 NaN/OOM/磁盘告警。
- E0 首轮已保存 `runs/E0-01/checkpoints/epoch=0-step=1988.ckpt`（SHA256 `117c3684a9d9d3068cb0554f6cd279691f988254352ad3322c609306f1563d87`）和 `last.ckpt`（SHA256 `50b32cd173836000a274cdfc18d5ca8f6eb1113e9b0caceff654dc0e3efa7d02`）。`last.ckpt` 内含 epoch 0、global step 1988、Adam optimizer 状态及 Lightning loop 状态。
- E0 首轮验证 `val/loss_lm=6.486662864685059`，有效码 **124/128**、Perplexity **97.3742904663086**。验证情感结果已按 406 条 trial、40 个独立句子保存；E0 情感头未训练，分类分数仅供诊断。
- 三组各自输出为 `runs/E0-01`、`runs/E1-01`、`runs/E2-01`；各自训练日志为 `logs/E0-01.log` 等。每组保留验证 LM Loss 最佳和最新可恢复 checkpoint。
- 后台监控每 30 秒记录 epoch、global step、各 loss、GPU 显存/利用率及剩余磁盘到 `monitor/status.json`、`monitor/E*_samples.jsonl`。非有限 loss、明确 OOM 或低磁盘会保存 incident JSON 并停止当前实验和后续队列。现有代码不暴露逐步 gradient norm；无法无侵入地监控所有有限值的梯度异常。
- E0/E1/E2 各自结束后会自动生成 `training_report.json`、`training_report.md`、`training_curve.csv` 和逐轮 `metrics/val_epoch=*.json`；三组全部结束后自动生成 `EPA_final_comparison.md`。训练成功或失败均不会覆盖历史实验目录。

源数据、文本模型、发布 GLIM checkpoint、V3 原始 VQ 初始化、split manifest 和 EPA 原型均按预检 SHA/大小核验，未删除或覆盖。清理审计见 `cleanup_candidates.md`、`cleanup_summary.md`。
