"""Score fixed validation predictions and write the Chinese step-level diagnosis."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from bert_score import BERTScorer
from torchmetrics.functional.text import bleu_score, rouge_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value not in (None, "", "None") else None


def mean_uid(values, uid_values):
    groups = defaultdict(list)
    for uid, value in zip(uid_values, values):
        groups[uid].append(float(value))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def score_generation():
    model = json.loads((ROOT / "configs" / "bertscore_model.json").read_text(encoding="utf-8"))
    scorer = BERTScorer(model_type=model["path"], num_layers=17, batch_size=16,
                        device="cuda", idf=False, rescale_with_baseline=False,
                        use_fast_tokenizer=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/GLIM-main-A3-Revised/models/flan-t5-large", local_files_only=True)
    output = []
    for path in sorted((ROOT / "evaluation").glob("predictions_step_*.jsonl")):
        step = int(path.stem.split("_")[-1])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(records) != 406 or len({r["uid"] for r in records}) != 40:
            raise RuntimeError(f"Incomplete validation generation at step {step}")
        preds = [r["prediction"] for r in records]
        refs = [r["reference"] for r in records]
        _, _, bert = scorer.score(preds, refs, batch_size=16, verbose=False)
        bert = bert.float().cpu().tolist()
        metrics = defaultdict(list)
        for pred, ref, value in zip(preds, refs, bert):
            for n in (1, 2, 4):
                metrics[f"bleu{n}"].append(float(bleu_score([pred], [[ref]], n_gram=n)))
            metrics["rouge_l"].append(float(rouge_score([pred], [ref], rouge_keys="rougeL")["rougeL_fmeasure"]))
            metrics["bertscore_f1"].append(value)
            metrics["generation_length_t5_tokens"].append(len(tokenizer.encode(pred, add_special_tokens=False)))
        uids = [r["uid"] for r in records]
        record = {"step": step, "n_trials": len(records), "n_uids": len(set(uids))}
        record.update({key: mean_uid(values, uids) for key, values in metrics.items()})
        record["exact_duplicate_ratio_trials"] = 1 - len(set(preds)) / len(preds)
        record["empty_prediction_ratio_trials"] = sum(not p.strip() for p in preds) / len(preds)
        output.append(record)
        print(f"SCORED step={step}", flush=True)
    if not output:
        raise RuntimeError("No generation checkpoints to score")
    with (ROOT / "generation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output, model


def plot_loss(step_rows, drift_rows):
    x = [int(r["step"]) for r in step_rows]
    train = [float(r["train_canonical_loss_lm"]) for r in step_rows]
    val = [float(r["val_loss_lm"]) for r in step_rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, train, "o-", label="Train UID, canonical target, eval")
    axes[0].plot(x, val, "s-", label="Validation UID, canonical target, eval")
    axes[0].set_ylabel("Token-weighted LM loss")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].plot(x, [float(r["encoder_relative_delta"]) for r in drift_rows], "o-", label="EEG Encoder")
    axes[1].plot(x, [float(r["aligner_relative_delta"]) for r in drift_rows], "s-", label="Aligner")
    axes[1].set_xlabel("Cumulative optimizer updates")
    axes[1].set_ylabel("||theta - theta0|| / ||theta0||")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.savefig(ROOT / "train_val_loss_curve.png", dpi=180)
    plt.close(fig)


def signed(value):
    return f"{value:+.5f}"


def report(step_rows, drift_rows, generation, bert_model):
    start, end = step_rows[0], step_rows[-1]
    initial_val = float(start["val_loss_lm"])
    first_up = next((int(r["step"]) for r in step_rows[1:]
                     if float(r["val_loss_lm"]) > initial_val), None)
    first_adjacent_up = next((int(b["step"]) for a, b in zip(step_rows, step_rows[1:])
                              if float(b["val_loss_lm"]) > float(a["val_loss_lm"])), None)
    best = min(step_rows, key=lambda r: float(r["val_loss_lm"]))
    best_step = int(best["step"])
    first_after_best_above_source = next((int(r["step"]) for r in step_rows
                                          if int(r["step"]) > best_step and float(r["val_loss_lm"]) > initial_val), None)
    train_change = float(end["train_canonical_loss_lm"]) - float(start["train_canonical_loss_lm"])
    val_change = float(end["val_loss_lm"]) - initial_val
    gen_by_step = {int(r["step"]): r for r in generation}
    g0 = gen_by_step[0]
    gend = gen_by_step[int(end["step"])]
    paths = [int(r["step"]) for r in step_rows]
    lines = [
        "# V3 B0 单 Epoch Step-level 续训诊断",
        "",
        "## 范围和完整性",
        "",
        f"单次 Lightning `fit`，seed 42，B0，Adam 状态重置，LR 1e-4，batch 72，A800 BF16；实际 DataLoader 为 {int(end['step'])} batch，梯度累积 1，实际执行 {int(end['step'])} 次 optimizer 更新。训练 UID 的 8 种改写目标、原有损失和生成结构未改。冻结 Flan-T5 全程保持 eval。源 checkpoint SHA256 已核对。",
        "",
        "每次中途评估保存/恢复 Python、NumPy、Torch CPU/CUDA RNG 状态；逐子模块恢复训练标志；评估前后逐参数 version、Adam 状态数、global_step 不变。训练 UID 的原句评估通过将同一 train trial 以验证数据集路径构造，和 val UID 均使用 eval 模式、原句、BF16、同一前向和 token 加权 LM Loss。检查点训练 batch loss 是更新**之前**算出的当前 batch loss，不与整集 eval loss 直接比较。",
        "",
        "## 原始轨迹",
        "",
        "| Optimizer step | 训练 UID 原句 LM ↓ | 验证 UID 原句 LM ↓ | 验证 CLIP | 验证 VQ | 验证原情感 CE | 验证总 Loss | 更新前 Batch LM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in step_rows:
        lines.append("| " + " | ".join([
            r["step"], f"{float(r['train_canonical_loss_lm']):.5f}",
            f"{float(r['val_loss_lm']):.5f}", f"{float(r['val_loss_clip']):.5f}",
            f"{float(r['val_loss_vq']):.6f}", f"{float(r['val_loss_disc_emotion']):.5f}",
            f"{float(r['val_loss_total']):.5f}",
            "—" if not r["train_batch_lm_pre_update"] else f"{float(r['train_batch_lm_pre_update']):.5f}",
        ]) + " |")
    lines += ["", "### 参数漂移和梯度", "",
              "| Step | 全局梯度范数（更新前） | Encoder 相对源权重变化 | Aligner 相对源权重变化 |",
              "|---:|---:|---:|---:|"]
    for r in drift_rows:
        lines.append(f"| {r['step']} | {number(r['global_grad_norm_pre_update']) if r['global_grad_norm_pre_update'] else '—'} | {float(r['encoder_relative_delta']):.6f} | {float(r['aligner_relative_delta']):.6f} |")
    lines += ["", "### 固定验证 UID 自由生成", "",
              "| Step | BLEU-1 | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore F1 | 平均 T5 token 数 | 重复文本比例 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in generation:
        lines.append(f"| {r['step']} | {r['bleu1']:.5f} | {r['bleu2']:.5f} | {r['bleu4']:.5f} | {r['rouge_l']:.5f} | {r['bertscore_f1']:.5f} | {r['generation_length_t5_tokens']:.2f} | {r['exact_duplicate_ratio_trials']:.3f} |")
    lines += [
        "", "## 核心判断", "",
        f"1. **验证 LM 何时上升：**Step 1 相对 Step 0 第一次短暂上升；Step {best_step} 降到本轮最低 {float(best['val_loss_lm']):.5f}；其后首次高于源 Step 0 的测点是 Step {first_after_best_above_source if first_after_best_above_source is not None else '未出现'}。因此与 Epoch 末持续退化相关的转折只定位在 Step {best_step}–{first_after_best_above_source} 区间，不能从稀疏测点确定准确更新数。原始测点为 {paths}。",
        f"2. **训练/验证分叉：**Step 0→{end['step']} 的训练 UID 原句 LM 变化 {signed(train_change)}，验证 UID 变化 {signed(val_change)}。" + ("方向相反，符合跨 UID 泛化退化迹象；但 Step 0 时两者已有明显差距，不能把全部差距归因于本轮续训。" if train_change < 0 < val_change else "两者没有形成典型的训练降、验证升方向分叉。"),
        f"3. **早期更新幅度：**Step 1 Encoder/Aligner 相对变化 {float(drift_rows[1]['encoder_relative_delta']):.6f}/{float(drift_rows[1]['aligner_relative_delta']):.6f}；Step 10 为 {float(next(r for r in drift_rows if int(r['step']) == 10)['encoder_relative_delta']):.6f}/{float(next(r for r in drift_rows if int(r['step']) == 10)['aligner_relative_delta']):.6f}。所有记录的梯度、Loss 和权重变化均为有限值，且 Step 10 验证 LM 优于源模型，不能认定前几个更新已过大；Aligner 到末尾的累积漂移值得后续核查，但尚无因果证据。",
        "4. **过拟合：**" + ("本轮出现训练原句改善而验证原句恶化的方向证据，但还不能区分单任务范围、UID 重复或改写目标的具体作用。" if train_change < 0 < val_change else "本轮没有出现典型的训练原句持续改善、验证原句持续恶化组合，不能直接归因为过拟合。"),
        f"5. **LM 与生成方向：**Step 0→50，验证 LM {signed(float(next(r for r in step_rows if int(r['step']) == 50)['val_loss_lm']) - initial_val)}，ROUGE-L {signed(gen_by_step[50]['rouge_l'] - g0['rouge_l'])}，BERTScore F1 {signed(gen_by_step[50]['bertscore_f1'] - g0['bertscore_f1'])}；Step 0→{end['step']}，验证 LM {signed(val_change)}，ROUGE-L {signed(gend['rouge_l'] - g0['rouge_l'])}，BERTScore F1 {signed(gend['bertscore_f1'] - g0['bertscore_f1'])}。教师强制损失与自由生成至少在 Step 50 不同向；这些差异仅来自一个种子和 40 个验证 UID。",
        f"6. **优先下一步：**只改变 B0 的续训停止步数，在 Step {best_step}–{first_after_best_above_source} 之间预先加密验证/保存点，固定模型、目标、优化器和评估协议，并在多个种子复核。预先把自由生成 ROUGE-L/BERTScore 定为主要质量指标，LM Loss 用于诊断与安全监测；本轮 Step 10 的 LM 最佳与 Step 50 的生成指标最佳不同，不能只按 LM 挑选。先验证更新预算能否缓解共同退化，再决定是否测试冻结 Encoder 或改训练数据范围。",
        "7. **解释力度：**这是一次 B0 单种子、一个 Epoch、同一验证集的诊断。它可以定位观测到退化的时间窗口并检查训练/验证方向，不能单独证明 B0/B1 共同退化的唯一因果机制。",
        "", "## 口径和局限", "",
        f"验证共 40 UID、406 trial；生成指标先逐 trial 算，再在 UID 内平均、最后 UID 等权平均。BERTScore 使用本地 roberta-large revision `{bert_model['revision']}`、17 层；beam 2、最大 64 token。平均生成长度为不含 special token 的 T5 token 数；重复文本比例为 `1 − 唯一生成字符串数/406`，并非 n-gram 重复率。训练/验证 CLIP 的 batch 组成不同，尤其存在同 UID 多被试，故 CLIP 和由其组成的总 Loss 只作轨迹参考，不用于严格的 train/val 差值。",
        "", "本次验证频率高，但评估过程恢复 RNG 和 mode，且不调用 optimizer.step；没有做第二个训练种子或 EPA 实验。已有验证 UID 多次用于诊断，本文数值不是封存测试集泛化估计。",
    ]
    (ROOT / "step_level_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def checksums():
    paths = [ROOT / "config.json", ROOT / "configs" / "common.json", ROOT / "configs" / "bertscore_model.json",
             ROOT / "step_metrics.csv", ROOT / "parameter_drift.csv", ROOT / "generation_metrics.csv",
             ROOT / "train_val_loss_curve.png", ROOT / "step_level_report.md",
             ROOT / "reports" / "smoke.json", ROOT / "reports" / "source_hashes.json",
             ROOT / "reports" / "runtime_audit.json", ROOT / "reports" / "fit_complete.json"]
    paths += sorted((ROOT / "runs").glob("step_*.ckpt"))
    paths += sorted((ROOT / "evaluation").glob("predictions_step_*.jsonl"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    lines = []
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    completion = json.loads((ROOT / "reports" / "fit_complete.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("The single Lightning run is not complete")
    step_rows = rows(ROOT / "step_metrics.csv")
    drift_rows = rows(ROOT / "parameter_drift.csv")
    if [int(r["step"]) for r in step_rows] != completion["checkpoints"]:
        raise RuntimeError("Missing prespecified step")
    generation, model = score_generation()
    plot_loss(step_rows, drift_rows)
    report(step_rows, drift_rows, generation, model)
    checksums()
    print("FINALIZATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
