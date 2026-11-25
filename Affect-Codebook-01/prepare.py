"""Build frozen EEG-only inputs once, from the exact V3 and approved split."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/root/autodl-tmp/Affect-Supervision-01")
V3 = Path("/root/autodl-tmp/GLIM-main-A3-v2")
sys.path.insert(0, str(V3))
from data.datamodule import ZuCoDataset
from model.glim_multitask import GLIMMultitask

CFG = json.loads((ASSETS / "step_level_diagnosis_run01/config.json").read_text())
GEN = CFG["generation"]
CLASSES = ["negative", "neutral", "positive"]


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def base_model():
    ck = CFG["checkpoint"]
    assert digest(ck["path"]) == ck["sha256"]
    package = torch.load(ck["path"], map_location="cpu", weights_only=False)
    hp = package["hyper_parameters"]
    assert (package["epoch"], package["global_step"], hp["variant"], hp["codebook_size"]) == (46, 93436, "A3-full", 128)
    model = GLIMMultitask(**hp)
    model.load_state_dict(package["state_dict"], strict=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = CFG["assets"]["text_model"]
    model.cuda()
    model.setup("fit")
    return model.eval().requires_grad_(False)


def datasets():
    assets = CFG["assets"]
    assert digest(assets["e1_manifest"]) == assets["e1_manifest_sha256"]
    manifest = pd.read_csv(assets["e1_manifest"])
    def relevant(df):
        return df.loc[df.dataset.eq("ZuCo1") & df.task.eq("task1") & df.phase.isin(["train", "val"])].copy()
    manifest = relevant(manifest)
    frame = relevant(pd.read_pickle(assets["data"]))
    keys = ["text uid", "phase", "input text"]
    m = manifest.set_index(keys, verify_integrity=True)
    counts = frame.groupby(keys).size().sort_index()
    pd.testing.assert_series_equal(counts, m["EEG rows"].sort_index(), check_names=False, check_dtype=False)
    expected = m["sentiment label"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
    if "sentiment label" in frame:
        present = frame["sentiment label"].notna()
        assert (frame.loc[present, "sentiment label"] == expected[present]).all()
    frame["sentiment label"] = expected
    assert set(expected) == set(CLASSES) and frame.groupby("text uid").phase.nunique().max() == 1
    for split, trials, uids in [("train",3316,320), ("val",406,40)]:
        part = frame.loc[frame.phase.eq(split)].copy()
        assert len(part) == trials and part["text uid"].nunique() == uids
        # The upstream val dataset mode selects the canonical sentence once per trial.
        part["phase"] = "val"
        yield split, ZuCoDataset(part, "val")


@torch.no_grad()
def generate(model, memory, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ids = model.text_model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                                        attention_mask=mask, do_sample=False, **GEN)
    return model.tokenizer.batch_decode(ids, skip_special_tokens=True)


@torch.no_grad()
def prepare():
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(42)
    model = base_model()
    audit = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, checkpoint=CFG["checkpoint"],
                 split_sha256=CFG["assets"]["e1_manifest_sha256"], generation=GEN, splits={})
    for split, ds in datasets():
        chunks = {k: [] for k in ("x", "memory", "mask", "original_logits", "targets", "uid", "label")}
        subjects, references = [], []
        for step, batch in enumerate(DataLoader(ds, batch_size=24)):
            batch = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            eeg, mask, prompts = batch["eeg"], batch["mask"], batch["prompt"]
            assert torch.isfinite(eeg).all() and mask.bool().any(1).all()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.p_embedder(model.p_embedder.encode(prompts,device=eeg.device), model.eval_pembed)
                hidden, _ = model.eeg_encoder(eeg, mask, p)
                mem, x = model.aligner.embed_eeg(hidden)
                mem, extra = model.condition_memory(hidden, mem, mask, batch["raw task key"], prompts[1])
                if step == 0:
                    native = model.shared_forward(batch)
                    assert torch.equal(native["eeg_embeds"], mem) and torch.equal(native["eeg_emb_vector"], x)
                    assert torch.equal(native["memory_mask"], extra["memory_mask"])
                    oracle = generate(model, native["eeg_embeds"][:4], native["memory_mask"][:4])
                    assert oracle == generate(model, mem[:4], extra["memory_mask"][:4])
            ids, _ = model.tokenize(batch["target text"], model.tgt_text_len)
            label = model.encode_labels(batch["sentiment label"]).long()
            assert torch.isfinite(mem).all() and torch.isfinite(x).all()
            values = (x, mem, extra["memory_mask"], extra["disc_logits"], ids, batch["text uid"], label)
            for key, value in zip(chunks, values):
                chunks[key].append(value.cpu())
            subjects.extend(prompts[2])
            references.extend(batch["raw input text"])
        cache = {k: torch.cat(v) for k,v in chunks.items()}
        cache.update(subject=subjects, reference=references)
        torch.save(cache, ROOT / (split + ".pt"))
        audit["splits"][split] = dict(trials=len(subjects), uids=int(cache["uid"].unique().numel()),
                                     memory_shape=list(cache["memory"].shape), features=list(cache["x"].shape),
                                     first_batch_native_memory_equal=True, native_generation_equal_n=4)
        print(split, audit["splits"][split],flush=True)
    write("preflight.json", audit)


if __name__ == "__main__":
    prepare()
