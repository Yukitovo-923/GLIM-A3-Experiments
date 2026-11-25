# 环境与运行说明

## 环境

原项目使用 Linux、Python 3.12、PyTorch/Lightning 和单 GPU。仓库提供的是历史部署依赖清单，不是完整锁文件；本次整理没有在新环境安装依赖或重新执行 GPU 实验。

以 A3 Revised 的依赖为起点，可在 Linux 上准备独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r GLIM-main-A3-Revised/requirements-a3.txt
```

GPU、驱动和 CUDA 需要与所选 PyTorch 构建兼容。历史训练入口使用 `bf16-mixed`；不同设备的精度支持需要单独核验。

## 外部资源

| 资源 | 用途 | 获取或准备方式 |
| --- | --- | --- |
| ZuCo EEG 数据及任务标签 | 数据加载、监督与评估 | 从数据提供方获取并按 GLIM 预处理流程生成；参见上游 README |
| 配对 EEG dataframe 和原始标签 dataframe | `datamodule.py`、标签审计 | 由预处理得到，历史文件名为 `zuco_eeg_label_8variants.df`、`zuco_label_8variants.df` |
| 本地 Flan-T5-large | 冻结语言模型与 tokenizer | 准备完整模型目录，通过 `--text-model-path` 或配置指定 |
| 发布版 GLIM checkpoint | A3 Revised 初始化 | 上游提供的 [checkpoint 页面](https://doi.org/10.6084/m9.figshare.29115161.v1) |
| V3 epoch 46 checkpoint | 后续情感监督、码本与探针实验 | 研究过程中生成的独立权重，需要另外取得 |
| Split manifest、情感原型、冻结特征 | 保持各实验的数据协议一致 | 使用对应实验流程生成，并核对配置中记录的来源与哈希 |

上游的数据预处理说明见 [GLIM](https://github.com/justin-xzliu/GLIM)。数据和模型均不随本仓库分发。

## 主项目的单批核验

在仓库根目录创建环境后，从项目目录执行，替换示例资源路径：

```bash
cd GLIM-main-A3-v2
python smoke_a3.py \
  --checkpoint /path/to/glim.ckpt \
  --data-path /path/to/zuco_eeg_label_8variants.df \
  --labels /path/to/zuco_label_8variants.df \
  --text-model-path /path/to/flan-t5-large \
  --config configs/a3_revised.json \
  --output runs/share-smoke-001
```

该脚本执行一批前向、反向和一次优化器更新，不进入完整训练循环。输出目录必须是新目录。

主训练入口为 `run_multitask.py fit`，需要与配置和资源匹配的真实数据 smoke 报告。具体参数可查看该脚本。VQ 初始化与 Pilot 则使用 `initialize_vq_v2.py` 和 `vq_pilot.py`；它们对应不同阶段，不应直接替代基础入口。

## 迁移历史实验

后续实验仍保留 `/root/autodl-tmp/...` 的绝对路径，包括 Python 常量、Shell 脚本和 JSON 配置。开始运行前，需要检查选定实验内的路径和缺失资源。例如：

```bash
rg -n '/root/autodl-tmp' Affect-Supervision-01 Affect-Codebook-01 SupCon-01
```

应把配置复制为本地版本后修改路径。保留哈希核验和句子划分审计；换用资源后，应明确记录来源和划分差异，不能把不同 checkpoint 或不同划分的结果视为同一实验。

历史监控、续训、队列和清理脚本面向原服务器环境，阅读时应先检查它们的目标目录和执行行为。只浏览代码不需要执行这些脚本。

## 本次整理检查

在仓库根目录执行 `python tools/check_snapshot.py`，可检查 Python 语法、JSON 格式、源码清单、常见凭据模式，以及 README 的本地链接。该检查只使用 Python 标准库，不加载模型或数据。
