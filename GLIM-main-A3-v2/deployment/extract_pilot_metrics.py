"""Read completed records from a W&B offline run without touching the writer."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def read_history(path):
    store = DataStore()
    store.open_for_scan(str(path))
    rows = []
    try:
        while True:
            try:
                data = store.scan_data()
            except (AssertionError, ValueError, OSError):
                break  # Live writer may have an unfinished final record.
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                key = "/".join(item.nested_key) if item.nested_key else item.key
                try:
                    row[key] = json.loads(item.value_json)
                except (ValueError, TypeError):
                    continue
            rows.append(row)
    finally:
        store.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_history(args.input)
    keys = sorted({key for row in rows for key in row})
    print("records", len(rows), "keys", len(keys))
    print("metric_keys", json.dumps([k for k in keys if any(t in k.lower() for t in
          ("epoch", "loss_lm", "macro_f1", "perplexity", "active_code", "bleu", "rouge", "gate"))]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
