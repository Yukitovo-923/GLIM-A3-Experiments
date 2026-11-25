"""Held-out descriptive geometry on the saved, unmodified sentence embeddings."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from text_space_audit import geometry, normalized, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    embeddings = np.load(run / "text_embeddings.npz")
    sentences = pd.read_csv(run / "sentences_official_labels.csv")
    if not np.array_equal(embeddings["uid"], sentences["text uid"].to_numpy()):
        raise ValueError("Sentence UID mismatch")
    result = {name: {phase: geometry(normalized(embeddings[name][sentences.phase == phase].astype(np.float64)),
                                     sentences.loc[sentences.phase == phase, "target"].to_numpy(int))
                     for phase in ("train", "val", "test")}
              for name in ("clip_text", "encoder_mean")}
    save_json(run / "geometry_by_split.json", result)
    print({name: {phase: value["cosine_silhouette"] for phase, value in splits.items()}
           for name, splits in result.items()})


if __name__ == "__main__":
    main()
