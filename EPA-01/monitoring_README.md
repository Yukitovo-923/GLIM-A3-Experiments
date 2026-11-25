# EPA-01 顺序训练与独立监控

启动日期：2026-11-08。后台顺序进程由 `setsid` 运行 `run_epa_sequence.sh`；当前/以后 PID 见 `logs/sequence.pid` 和 `logs/E0-01.pid`、`E1-01.pid`、`E2-01.pid`。唯一执行顺序 E0 → E1 → E2。每组正常结束并确认非空 `checkpoints/last.ckpt` 后，才进入下一组；失败即停止序列。

独立监控进程 PID 见 `logs/monitor.pid`。每 30 秒读取 W&B 离线历史、GPU 显存与利用率、数据盘可用空间；最新状态在 `monitor/status.json`，按组时间序列在 `monitor/E0_samples.jsonl` 等。监控只读取训练日志和指标，不修改模型、优化器、数据或正在运行的训练配置。

若记录到非有限 loss、训练日志出现 CUDA OOM，或数据盘可用空间低于 5 GiB，监控先写 `monitor/incident_*.json`，再向**核对过命令行身份**的顺序进程和当前训练进程发送 SIGTERM，阻止后续组启动。原日志及已有 checkpoint 保留。其他 Python 异常导致训练退出时，顺序脚本同样停止后续组。显存和磁盘状态属于实际采样值，不是预估。

限制：现有 Lightning 训练代码没有逐步记录 gradient norm；无需重启或改训练代码的独立监控无法直接读取另一个 Python 进程中的梯度。因此本监控可发现非有限 loss 和显式 autograd/优化器异常，但**不能承诺识别所有有限值的梯度爆炸**。本轮真实 smoke 已验证各分支初始梯度存在且有限。此限制不得在后续报告中写作“梯度异常监控全面通过”。

每组正常结束后，监控在对应 `runs/E*-01/` 写入 `training_report.json`、`training_report.md`、`training_curve.csv`，并保留逐轮 `metrics/val_epoch=*.json`。全部完成后生成 `/root/autodl-tmp/EPA-01/EPA_final_comparison.md`。E0 的情感头未训练，分类分数仅供诊断。
