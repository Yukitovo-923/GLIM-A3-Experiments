# EPA-01 改为三组各 10 epochs：切换记录

时间：2026-11-08（北京时间）。实验预算现为每组总计 epoch 0–9，共 10 epochs；顺序 E0 → E1 → E2，单张 A800 串行运行。原 GLIM/V3/A0、发布版 checkpoint、文本原型、数据和共同初始化未改动。

## E0 连续性

原 50-epoch E0 实验在 `runs/E0-01/` 完成 epoch 0 与 1 后停止，旧队列 PID 3675、训练 PID 3682 和旧监控 PID 5625 均已退出。旧进程已进入 epoch 2 的未完成部分，**这部分权重被舍弃**；已完成的训练/验证记录和 checkpoint 全部保留。

恢复来源：`/root/autodl-tmp/EPA-01/runs/E0-01/checkpoints/epoch=1-step=3976.ckpt`，SHA256 `b103118f9af77c314eccba4c283913469aabff5b1d1ca1d5e2ab9b357bc3972d`。已验证 epoch=1、global_step=3976、state_dict 214 项、Adam optimizer 状态 1 组和 Lightning loop 状态。新的 E0 运行从 epoch **2** 继续，目标总计 epoch **9**，不是追加 10 epochs。独立输出 `/root/autodl-tmp/EPA-01/runs/E0-10-02/`。

首次 10-epoch 恢复尝试保留在 `runs/E0-10-01/`，因原 GLIM 的 checkpoint 保存逻辑省略冻结 T5 的键，而 Lightning 严格恢复要求全键匹配而退出；**E1/E2 未被启动**。在 EPA 自有模型类中加入仅允许缺失 `text_model.*` 的恢复校验，其余所有键严格匹配。独立无训练恢复测试 PASS：恢复 214 个非 T5 参数键，冻结 T5 的 560 个键由已核验的本地 Flan-T5 加载，Adam 状态含 206 项。未修改 T5、GLIM 原始代码或任何历史 checkpoint。

## 当前独立 10-epoch 队列

- 共用初始化：`/root/autodl-tmp/EPA-01/initialization/common_init.pt`，SHA256 `b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e`。
- 训练入口：`run_epa_10.py --max-epochs 10`，只覆盖总 epoch 上限；模型结构、损失权重、学习率、batch size、BF16、数据划分均与预检一致。
- 队列：`run_epa_10_sequence.sh`，PID **7277**；E0 恢复训练 PID **7288**；独立监控 PID **8099**。SSH 断开后继续运行。E1/E2 仅在前一组成功结束、`last.ckpt` 内确认 epoch=9、global_step=19880、优化器与 loop 状态后启动。
- 新输出：`runs/E0-10-02/`、`runs/E1-10-02/`、`runs/E2-10-02/`。每组独立 `best`、`last`、训练日志、逐轮验证指标和最终报告。
- 队列日志：`logs/sequence10-r2.log`；各组日志 `logs/E0-10-02.log` 等；最新监控状态 `monitor10-r2/status.json`，异常现场 `monitor10-r2/incident_*.json`。
- 2026-11-08 14:38:35 采样：E0 epoch **2**、global step **4249**，GPU 显存 **23,083 MiB**，数据盘剩余 **20,119,482,368 bytes**。训练 loss 有限；没有监控告警。
- 正式三组完成后自动生成 `/root/autodl-tmp/EPA-01/EPA_pilot10_comparison_r2.md`；各组 `training_curve.csv`、`training_report.json`/`.md` 和 `metrics/val_epoch=*.json` 保留。

监控可依据日志阻止 NaN、明确 OOM 和低于 5 GiB 的磁盘空间；现有训练代码未记录逐步梯度范数，外部监控无法识别所有有限值梯度异常。E0 的最终曲线将合并原运行已完成的 epoch 0–1 与新运行的 epoch 2–9，舍弃旧运行未完成的 epoch 2 片段。
