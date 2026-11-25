"""Audit normalized ZuCo sentence identities; never reassign an existing phase."""
import ast
import json
import mmap
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"negative": 0, "neutral": 1, "positive": 2}


def mapped_step3(path):
    # Trusted protocol-5 STEP3 pickle. NumPy reconstructs arrays from views of
    # the large unframed BYTEARRAY8 payloads. Metadata and small masks use RAM.
    stream = path.open("rb")
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    class Reader(pickle._Unpickler):
        dispatch = pickle._Unpickler.dispatch.copy()

        def bytearray8(self):
            length, = struct.unpack("<Q", self.read(8))
            if length < 65536:
                self.append(bytearray(self.read(length)))
                return
            frame = self._unframer.current_frame
            if frame and frame.read(1):
                raise ValueError("Unsupported framed EEG buffer")
            start = stream.tell()
            stream.seek(length, 1)
            self.append(memoryview(mapping)[start:start + length])

        dispatch[pickle.BYTEARRAY8[0]] = bytearray8

    with stream:
        return Reader(stream).load()


def normalized_texts(texts):
    notebook = json.loads((ROOT / "data/__STEP4_eeg_text_match_split.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "".join(c["source"]).startswith("typobook ="))
    rules = ast.literal_eval(ast.parse(source).body[0].value)
    def revise(text):
        for old, new in rules.items():
            text = text.replace(old, new)
        return text
    return texts.map(revise)


def audit(frame, reference):
    frame = frame.copy(deep=False)
    keys = ["dataset", "task", "input text"]
    if "input text" not in frame:
        frame["input text"] = normalized_texts(frame["text"])
    ref = reference[keys + ["text uid", "sentiment label"]].drop_duplicates()
    if ref.duplicated(keys).any() or ref.groupby("text uid")["input text"].nunique().gt(1).any():
        raise ValueError("Ambiguous reference sentence")
    # Prefer the existing sentence identifier; validate its canonical text. Use
    # the STEP4-normalized input text only for records without an identifier.
    matched = ref.set_index(keys).reindex(pd.MultiIndex.from_frame(frame[keys])).reset_index()
    have_id = np.zeros(len(frame), dtype=bool)
    if "text uid" in frame:
        have_id = frame["text uid"].notna().to_numpy()
        id_keys = ["dataset", "task", "text uid"]
        identified = ref.set_index(id_keys).reindex(pd.MultiIndex.from_frame(frame.loc[have_id, id_keys]))
        if not np.array_equal(identified["input text"].to_numpy(), frame.loc[have_id, "input text"].to_numpy()):
            raise ValueError("Sentence ID and canonical text disagree")
        matched.loc[have_id, "text uid"] = frame.loc[have_id, "text uid"].to_numpy()
        matched.loc[have_id, "sentiment label"] = identified["sentiment label"].to_numpy()
    if matched["text uid"].isna().any():
        raise ValueError("Unmatched normalized sentence in EEG table")
    frame["text uid"] = matched["text uid"].to_numpy()
    expected = matched["sentiment label"].to_numpy(copy=True)
    sr = (frame["dataset"].eq("ZuCo1") & frame["task"].eq("task1")).to_numpy()
    expected[~sr] = np.nan
    if not set(pd.Series(expected).dropna()) <= CLASSES.keys():
        raise ValueError("Unknown SR label")
    if "sentiment label" in frame:
        given = frame["sentiment label"].fillna("<missing>").to_numpy()
        if np.any((given != "<missing>") & sr & (given != pd.Series(expected).fillna("<missing>").to_numpy())):
            raise ValueError("Existing SR sentiment conflicts with reference")
        frame["original sentiment label"] = frame["sentiment label"]
    else:
        frame["original sentiment label"] = expected
    frame["sentiment label"] = expected
    if "phase" not in frame:
        # Reproduce official STEP4 cell 10 on observed EEG sentence IDs.
        duplicates = set(reference.loc[reference["text uid"].duplicated(), "text uid"])
        phases = {}
        for _, group in frame.groupby(["dataset", "task"]):
            uids = list(set(group["text uid"].values))
            unique = [u for u in uids if u not in duplicates]
            a = len(unique)
            k = 0.8 - 0.2 * (len(uids) - a) / a
            splits = np.split(np.random.default_rng(42).permutation(unique),
                              [int(k*a), int((a+k*a)/2)])
            for name, part in zip(("train", "val", "test"), splits):
                phases.update(dict.fromkeys(part, name))
        phases.update(dict.fromkeys(duplicates, "train"))
        frame["phase"] = frame["text uid"].map(phases)
    if (set(frame["phase"]) != {"train","val","test"} or
            any(frame.groupby(k)["phase"].nunique().gt(1).any() for k in ("text uid","input text"))):
        raise ValueError("Invalid or leaking sentence split")
    manifest = frame[keys + ["text uid","phase","original sentiment label","sentiment label"]].copy()
    manifest["match method"] = np.where(have_id, "sentence ID + canonical text", "exact normalized text")
    raw_sr = reference.loc[reference.task.eq("task1") & reference.dataset.eq("ZuCo1"),
                           ["text uid", "raw label"]].drop_duplicates().set_index("text uid")["raw label"]
    manifest["source SR raw label"] = manifest["text uid"].map(raw_sr).where(sr)
    manifest["target"] = manifest["sentiment label"].map(CLASSES).fillna(-100).astype(int)
    manifest = manifest.groupby(list(manifest.columns), dropna=False).size().reset_index(name="EEG rows")
    return frame, manifest
