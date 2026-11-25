"""Plan and conservatively remove only verified regenerable Python bytecode."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
EPA = ROOT / "EPA-01"
CACHE_DIRS = (
    "GLIM-main-A3-Revised/deployment/__pycache__",
    "GLIM-main-A3-Revised/model/__pycache__",
    "GLIM-main-A3-Revised/__pycache__",
    "GLIM-main-A3-v2/model/__pycache__",
    "GLIM-main-A3-v2/data/__pycache__",
    "GLIM-main-A3-v2/deployment/__pycache__",
    "GLIM-main-A3-v2/__pycache__",
    "Text-space-Audit/__pycache__",
    "EPA-01/__pycache__",
)
PLAN = EPA / "cleanup_plan.json"
REPORT = EPA / "cleanup_candidates.md"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_source(path):
    if path.suffix != ".pyc" or ".cpython-" not in path.name:
        return None
    if path.is_symlink() or path.parent.is_symlink() or ".venv" in path.parts:
        return None
    if path.parent.name != "__pycache__":
        return None
    source = path.parent.parent / (path.name.split(".cpython-")[0] + ".py")
    if not source.is_file() or source.is_symlink():
        return None
    if not path.resolve().is_relative_to(ROOT.resolve()):
        return None
    return source


def open_paths():
    opened = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    opened.add(os.path.realpath(fd))
                except OSError:
                    continue
        except (OSError, PermissionError):
            continue
    return opened


def snapshot():
    files = []
    opened = open_paths()
    for suffix in CACHE_DIRS:
        directory = ROOT / suffix
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.iterdir()):
            source = safe_source(path)
            if source is None or str(path.resolve()) in opened:
                continue
            stat = path.stat()
            files.append({"path": str(path), "source": str(source),
                          "size": stat.st_size, "allocated": stat.st_blocks * 512,
                          "dev": stat.st_dev, "inode": stat.st_ino,
                          "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)})
    return files


def write_report(files, executed=None):
    total = sum(row["size"] for row in files)
    allocated = sum(row["allocated"] for row in files)
    lines = ["# EPA-01 数据盘清理候选", "",
             "本清单只允许删除下列已核验的 Python 字节码文件；对应 `.py` 源码存在，可自动重建。",
             "未修改虚拟环境、原始代码、数据、权重、日志、manifest 或报告。",
             f"候选文件：{len(files)}；逻辑大小：{total} bytes；实际分配块：{allocated} bytes。", "",
             "| 完整路径 | 大小（bytes） | 理由 | 可重建 | 其他依赖 |",
             "|---|---:|---|---|---|"]
    for row in files:
        lines.append(f"| `{row['path']}` | {row['size']} | Python 编译缓存，源文件 `{row['source']}` 存在 | 是，重新 import 后生成 | 无持久依赖；计划时未被进程打开 |")
    lines += ["", "## 受保护的大文件及目录：未列入自动删除", "",
              "- `/root/autodl-tmp/GLIM-main-A3-Revised/data/tmp/zuco_eeg_label_8variants.df`：约 14 GB，真实 EEG 数据。",
              "- `/root/autodl-tmp/GLIM-main-A3-Revised/models/flan-t5-large/`：约 3.0 GB，T5 权重。",
              "- `/root/autodl-tmp/A3-v2-V3-50-A800/run-02/checkpoints/`：约 9.4 GB，V3-Long 历史 checkpoint。",
              "- `/root/autodl-tmp/GLIM-main-A3-v2/runs/A3-v2-pilot-5-r2/`：约 1.4 GB，V3 Pilot 历史结果。",
              "- `/root/autodl-tmp/GLIM-main-A3-Revised/runs/A3-revised-train-01/`：约 1.7 GB，A3-R 历史结果。",
              "- `/root/autodl-tmp/EPA-01/initialization/common_init.pt`：EPA 共同初始化，必须保留。",
              "上述 checkpoint、实验目录、数据及权重即使有重复，均需用户另行确认；本轮不删除。", "",
              "## 其他缓存和临时文件", "",
              "`/root/.cache` 约 40 KB，且不在数据盘；未清理。未发现数据盘上的 `.part`、`.partial`、`.tmp`、`.download` 或 `.incomplete` 文件（虚拟环境已排除）。", ""]
    if executed is not None:
        lines += ["## 实际执行", "", f"成功删除 {len(executed)} 个 `.pyc` 文件。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "execute"))
    args = parser.parse_args()
    if args.mode == "plan":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        files = snapshot()
        PLAN.write_text(json.dumps(files, indent=2), encoding="utf8")
        write_report(files)
        print(json.dumps({"status": "PLANNED", "files": len(files),
                          "bytes": sum(row["size"] for row in files),
                          "allocated_bytes": sum(row["allocated"] for row in files),
                          "report": str(REPORT)}))
        return
    files = json.loads(PLAN.read_text(encoding="utf8"))
    now = {row["path"]: row for row in snapshot()}
    if len(now) != len(files) or any(now.get(row["path"]) != row for row in files):
        raise RuntimeError("Cache candidate changed or is now in use; nothing deleted")
    for row in files:
        path = Path(row["path"])
        if safe_source(path) is None:
            raise RuntimeError(f"Unsafe candidate: {path}")
    deleted = []
    for row in files:
        Path(row["path"]).unlink()
        deleted.append(row["path"])
    write_report(files, deleted)
    (EPA / "cleanup_result.json").write_text(json.dumps({
        "status": "PASS", "files_deleted": len(deleted),
        "logical_bytes": sum(row["size"] for row in files),
        "allocated_bytes": sum(row["allocated"] for row in files),
        "paths": deleted}, indent=2), encoding="utf8")
    print(json.dumps({"status": "PASS", "files_deleted": len(deleted),
                      "allocated_bytes": sum(row["allocated"] for row in files)}))


if __name__ == "__main__":
    main()
