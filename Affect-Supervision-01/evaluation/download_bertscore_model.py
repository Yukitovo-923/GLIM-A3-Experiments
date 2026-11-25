"""Fetch a pinned roberta-large model into this experiment's evaluation directory."""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "bertscore_model.json"
CACHE = ROOT / "evaluation" / "hf_cache"
MODEL = "roberta-large"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite existing BERTScore model record")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(CACHE))
    revision = HfApi(endpoint=os.environ["HF_ENDPOINT"]).model_info(MODEL).sha
    path = Path(snapshot_download(
        repo_id=MODEL,
        revision=revision,
        cache_dir=CACHE,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt",
                        "special_tokens_map.json"],
        max_workers=2,
    ))
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    record = {"model": MODEL, "revision": revision, "path": str(path),
              "weights_sha256": sha256(weights), "tokenizer_files": sorted(
                  item.name for item in path.iterdir() if item.is_file() and item.name != weights.name)}
    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
