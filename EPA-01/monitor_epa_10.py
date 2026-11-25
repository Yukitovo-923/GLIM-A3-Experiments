"""Read-only EPA metric/GPU/disk monitor with stop-on-confirmed-failure safeguards."""
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from deployment.extract_pilot_metrics import read_history

EPA = Path("/root/autodl-tmp/EPA-01")
MON = EPA / "monitor10-r2"
ARMS = ("E0", "E1", "E2")
MIN_FREE = 5 * 1024**3


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pid_matches(pid, needle):
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return needle in cmd
    except OSError:
        return False


def gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    memory, util = (int(x.strip()) for x in result.stdout.strip().splitlines()[0].split(","))
    return {"memory_mib": memory, "utilization_percent": util}


def history(arm):
    paths = list((EPA / "runs" / f"{arm}-10-02" / "wandb").glob("offline-run-*/run-*.wandb"))
    if len(paths) != 1:
        return []
    try:
        return read_history(paths[0])
    except (OSError, ValueError, AssertionError):
        return []


def completed_e0_history_before_resume():
    paths = list((EPA / "runs" / "E0-01" / "wandb").glob("offline-run-*/run-*.wandb"))
    if len(paths) != 1:
        return []
    return [row for row in read_history(paths[0]) if row.get("epoch") in (0, 1)]


def current_arm():
    for arm in ARMS:
        path = EPA / "logs" / f"{arm}-10-02.pid"
        if path.exists():
            pid = int(path.read_text().strip())
            if pid_matches(pid, f"config_{arm}.json") and pid_matches(pid, "run_epa_10.py fit"):
                return arm, pid
    return None, None


def append_json(path, row):
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def incident(reason, arm, pid, snapshot):
    record = {"time": now(), "reason": reason, "arm": arm, "pid": pid,
              "snapshot": snapshot, "action": "SIGTERM to verified EPA sequence and trainer; later arms blocked"}
    path = MON / f"incident_{int(time.time())}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    sequence_pid = int((EPA / "logs" / "sequence10-r2.pid").read_text().strip())
    if pid_matches(sequence_pid, "run_epa_10_sequence.sh"):
        os.kill(sequence_pid, signal.SIGTERM)
    if pid_matches(pid, f"config_{arm}.json") and pid_matches(pid, "run_epa_10.py fit"):
        os.kill(pid, signal.SIGTERM)
    return path


def flatten_epoch_rows(rows):
    epochs = {}
    for row in rows:
        epoch = row.get("epoch")
        if not isinstance(epoch, int):
            continue
        epochs.setdefault(epoch, {}).update(row)
    return [epochs[key] for key in sorted(epochs)]


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def report_arm(arm, completed):
    run = EPA / "runs" / f"{arm}-10-02"
    if not run.is_dir() or (run / "training_report.json").exists():
        return
    rows = history(arm)
    if arm == "E0":
        rows = completed_e0_history_before_resume() + rows
    epochs = flatten_epoch_rows(rows)
    write_csv(run / "training_curve.csv", epochs)
    last = run / "checkpoints" / "last.ckpt"
    bests = sorted((run / "checkpoints").glob("epoch=*.ckpt"))
    best = bests[0] if len(bests) == 1 else None
    vals = sorted((run / "metrics").glob("val_epoch=*.json"))
    val = read_json(vals[-1]) if vals else None
    head = "epa" if arm == "E2" else "original_emotion_head"
    selected = val.get("heads", {}).get(head, {}) if val else {}
    final = epochs[-1] if epochs else {}
    checkpoint = lambda p: {"path": str(p), "bytes": p.stat().st_size,
                            "sha256": digest(p)} if p and p.is_file() else None
    report = {
        "arm": arm, "status": "COMPLETE" if completed else "FAILED_OR_INCOMPLETE",
        "resumed_from": str(EPA / "runs/E0-01/checkpoints/epoch=1-step=3976.ckpt") if arm == "E0" else None,
        "time": now(), "epochs_logged": len(epochs),
        "final_epoch_logged": final.get("epoch"),
        "final_global_step_logged": final.get("trainer/global_step"),
        "final_train_lm_loss": final.get("train/loss_lm"),
        "final_val_lm_loss": final.get("val/loss_lm"),
        "val_active_code_ratio": final.get("val/active_code_ratio"),
        "val_active_code_count": (round(128 * final["val/active_code_ratio"])
                                  if isinstance(final.get("val/active_code_ratio"), (int, float)) else None),
        "val_codebook_perplexity": final.get("val/codebook_perplexity"),
        "emotion_head_evaluated": head,
        "emotion_head_supervised": arm != "E0",
        "final_validation_trial": selected.get("trial_level"),
        "final_validation_sentence": selected.get("sentence_level"),
        "last_checkpoint": checkpoint(last), "best_checkpoint": checkpoint(best),
        "validation_metrics_file": str(vals[-1]) if vals else None,
        "training_log": str(EPA / "logs" / f"{arm}-10-02.log"),
        "curve_csv": str(run / "training_curve.csv"),
    }
    (run / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    sentence = selected.get("sentence_level", {})
    text = [f"# EPA-01 {arm} 训练报告", "",
            f"状态：{report['status']}；最终记录 epoch {report['final_epoch_logged']}，global step {report['final_global_step_logged']}。",
            f"验证 LM Loss：{report['final_val_lm_loss']}；验证句子级 Macro-F1：{sentence.get('macro_f1')}；Balanced Accuracy：{sentence.get('balanced_accuracy')}。",
            f"有效码：{report['val_active_code_count']}/128；Perplexity：{report['val_codebook_perplexity']}。",
            f"情感头：{head}；是否接受情感监督：{arm != 'E0'}。",
            f"最后 checkpoint：{last if last.exists() else '缺失'}；最佳 checkpoint：{best if best else '未能唯一识别'}。",
            f"完整曲线：{run / 'training_curve.csv'}；逐轮验证：{run / 'metrics'}。", ""]
    (run / "training_report.md").write_text("\n".join(text), encoding="utf8")


def final_report():
    if not all((EPA / "runs" / f"{arm}-10-02" / "training_report.json").exists() for arm in ARMS):
        return
    reports = [read_json(EPA / "runs" / f"{arm}-10-02" / "training_report.json") for arm in ARMS]
    lines = ["# EPA-01 E0/E1/E2 训练汇总", "",
             "| 实验 | 状态 | 最终 epoch | Global step | 验证 LM Loss | 句子级 Macro-F1 | Balanced Accuracy | 有效码 | Perplexity |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        sent = r.get("final_validation_sentence") or {}
        lines.append(f"| {r['arm']} | {r['status']} | {r['final_epoch_logged']} | {r['final_global_step_logged']} | {r['final_val_lm_loss']} | {sent.get('macro_f1')} | {sent.get('balanced_accuracy')} | {r['val_active_code_count']} | {r['val_codebook_perplexity']} |")
    lines += ["", "E0 的原情感头未受情感监督，其分类分数仅供诊断。",
              "E1 与 E2 的监督作用层级不同，比较的是整体方案，不能单独归因于文本原型目标。",
              "各组逐轮曲线、验证混淆矩阵、逐被试结果、best/last SHA256 见独立 run 目录。", ""]
    (EPA / "EPA_pilot10_comparison_r2.md").write_text("\n".join(lines), encoding="utf8")


def main():
    MON.mkdir(exist_ok=True)
    seen_step = {}
    reported = set()
    while True:
        arm, pid = current_arm()
        free = shutil.disk_usage(EPA).free
        snapshot = {"time": now(), "active_arm": arm, "pid": pid, "disk_free_bytes": free}
        try:
            snapshot.update(gpu())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            snapshot["gpu_query_error"] = repr(exc)
        rows = history(arm) if arm else []
        latest = next((row for row in reversed(rows) if isinstance(row.get("trainer/global_step"), int)), {})
        snapshot["epoch"] = latest.get("epoch")
        snapshot["global_step"] = latest.get("trainer/global_step")
        snapshot["losses"] = {k: v for k, v in latest.items() if "loss" in k}
        (MON / "status.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf8")
        if arm and latest.get("trainer/global_step") != seen_step.get(arm):
            append_json(MON / f"{arm}_samples.jsonl", snapshot)
            seen_step[arm] = latest.get("trainer/global_step")
        if arm:
            nonfinite = []
            for row in rows:
                for key, value in row.items():
                    if "loss" not in key:
                        continue
                    if (value is None or
                        (isinstance(value, (int, float)) and not math.isfinite(value)) or
                        (isinstance(value, str) and value.lower() in
                         ("nan", "inf", "-inf", "infinity", "-infinity"))):
                        nonfinite.append(f"step={row.get('trainer/global_step')}:{key}")
            log = EPA / "logs" / f"{arm}-10-02.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 65536))
                    tail = stream.read().decode(errors="replace")
            else:
                tail = ""
            oom = bool(re.search(r"CUDA out of memory|OutOfMemoryError", tail, re.I))
            if nonfinite or free < MIN_FREE or oom:
                reason = f"nonfinite losses: {nonfinite}" if nonfinite else ("disk below 5 GiB" if free < MIN_FREE else "OOM log evidence")
                path = incident(reason, arm, pid, snapshot)
                print(f"{now()} STOP {arm}: {reason}; incident={path}", flush=True)
                return
        sequence = (EPA / "logs" / "sequence10-r2.log").read_text(encoding="utf8", errors="replace")
        for completed_arm in ARMS:
            if completed_arm not in reported and re.search(rf"\bCOMPLETE {completed_arm};", sequence):
                report_arm(completed_arm, completed=True)
                reported.add(completed_arm)
        sequence_pid = int((EPA / "logs" / "sequence10-r2.pid").read_text().strip())
        if not pid_matches(sequence_pid, "run_epa_10_sequence.sh") and arm is None:
            for pending in ARMS:
                if pending not in reported and (EPA / "runs" / f"{pending}-01").is_dir():
                    report_arm(pending, completed=False)
            final_report()
            print(f"{now()} sequence ended; monitor exit", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()


