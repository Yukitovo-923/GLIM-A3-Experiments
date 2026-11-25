# EPA-01 数据盘安全清理结果

日期：2026-11-08。目标文件系统 `/dev/md127`，挂载于 `/root/autodl-tmp`。

| 项目 | 清理前 | 清理后 |
|---|---:|---:|
| 可用空间 | 20,608,442,368 bytes | 20,609,036,288 bytes |
| 已用空间 | 33,078,648,832 bytes | 33,078,054,912 bytes |
| 实际净释放 | — | **593,920 bytes（约 0.57 MiB）** |
| 删除文件 | — | **41 个** |

仅删除了 `cleanup_candidates.md` 逐项列出的 `.pyc` Python 编译缓存。原 `.py` 源码均存在，计划和执行时未被进程打开；没有删除任何目录、checkpoint、数据、模型权重、虚拟环境、manifest、报告或重要日志。pip/torch 缓存及中断下载在数据盘上没有发现可观的安全候选，故未清理。41 个删除对象原先占 589,824 bytes 分配块；`df` 的净变化为 593,920 bytes，两者相差一个 4 KiB 文件系统块。

受保护文件核验：

- EPA `common_init.pt` SHA256：`b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e`，前后相同。
- 无前缀情感原型 SHA256：`ceaaccee85eee260c8c6e540bf441fced6421b2912aa9014d246e0cd55f5f31d`，前后相同。
- E1 split manifest SHA256：`13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17`，前后相同。
- 发布版 GLIM checkpoint SHA256：`25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`，前后相同。
- 真实 EEG 配对数据 `zuco_eeg_label_8variants.df`：14,671,736,481 bytes，前后相同。
- Flan-T5 `model.safetensors`：3,132,668,804 bytes，前后相同。

服务器保留 `/root/autodl-tmp/EPA-01/cleanup_candidates.md`、`cleanup_plan.json`、`cleanup_result.json` 作为完整审计记录。清理未获得大量空间，因为占用主要来自必须保留的真实数据、模型权重和历史 checkpoint。当前约 20.61 GB 可用；EPA 新入口仅保存各实验的最佳和最新 checkpoint。
