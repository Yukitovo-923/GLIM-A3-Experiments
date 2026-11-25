"""V3 A3-full with an optional EEG-side Emotion Prototype Alignment loss."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.glim_multitask import GLIMMultitask


ARMS = ("B0_Baseline", "B1_EPA")
CLASSES = ("negative", "neutral", "positive")


def weighted_epa_ce(logits, targets, uids, train_uid_counts):
    valid = targets.ge(0)
    if not valid.any():
        return logits.float().sum() * 0.0
    weights = torch.tensor(
        [1.0 / train_uid_counts[int(uid)] if int(uid) in train_uid_counts else 1.0
         for uid in uids.detach().cpu().tolist()],
        device=logits.device, dtype=torch.float32,
    ) * valid.float()
    per_trial = F.cross_entropy(logits.float(), targets.long().masked_fill(~valid, -100),
                                ignore_index=-100, reduction="none")
    return (per_trial * weights).sum() / weights.sum().clamp_min(1e-12)


class AffectGLIM(GLIMMultitask):
    def __init__(self, *, arm, prototype_path=None, epa_temperature=0.07,
                 epa_weight=0.2, **v3_config):
        if arm not in ARMS:
            raise ValueError(f"Invalid arm: {arm}")
        if v3_config.get("variant") != "A3-full":
            raise ValueError("This experiment requires the unchanged V3 A3-full generator")
        if v3_config.get("disc_emotion_weight") != 0.2:
            raise ValueError("Both arms must retain V3 discrete emotion CE weight 0.2")
        if epa_temperature <= 0 or epa_weight < 0:
            raise ValueError("Invalid EPA temperature or weight")
        super().__init__(**v3_config)
        if not hasattr(self, "attribute") or not hasattr(self, "disc_head"):
            raise RuntimeError("V3 attribute path or discrete sentiment head is missing")
        self.arm = arm
        self.epa_temperature = float(epa_temperature)
        self.epa_weight = float(epa_weight)
        self.train_uid_counts = None
        if arm == "B1_EPA":
            if prototype_path is None:
                raise ValueError("B1 needs frozen text prototypes")
            prototypes = torch.as_tensor(np.load(Path(prototype_path), allow_pickle=False),
                                         dtype=torch.float32)
            if prototypes.shape != (3, 1024) or not torch.isfinite(prototypes).all():
                raise ValueError("Expected three finite 1024-dimensional prototypes")
            torch.testing.assert_close(prototypes.norm(dim=1), torch.ones(3), atol=1e-5, rtol=0)
            self.register_buffer("epa_prototypes", prototypes.clone(), persistent=True)
            self.epa_projector = nn.Linear(1024, 1024)
        self.save_hyperparameters(dict(affect_arm=arm, epa_temperature=epa_temperature,
                                       epa_weight=epa_weight))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Allow only omitted frozen local T5 tensors when restoring a run checkpoint."""
        if strict and hasattr(self, "text_model") and not any(
                key.startswith("text_model.") for key in state_dict):
            expected = set(self.state_dict())
            supplied = set(state_dict)
            missing = expected - supplied
            unexpected = supplied - expected
            if (unexpected or not missing or
                    any(not key.startswith("text_model.") for key in missing)):
                raise RuntimeError(f"Invalid run checkpoint keys: missing={sorted(missing)}, "
                                   f"unexpected={sorted(unexpected)}")
            if any(param.requires_grad for param in self.text_model.parameters()):
                raise RuntimeError("Local T5 is not frozen during checkpoint restore")
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            if set(result.missing_keys) != missing or result.unexpected_keys:
                raise RuntimeError("Run checkpoint restore did not match the key audit")
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def on_fit_start(self):
        super().on_fit_start()
        if self.arm == "B1_EPA":
            counts = getattr(self.trainer.datamodule, "train_uid_counts", None)
            if not counts or len(counts) != 320:
                raise RuntimeError("B1 needs audited train UID trial counts")
            self.train_uid_counts = counts

    def shared_forward(self, batch):
        out = super().shared_forward(batch)
        if self.arm == "B1_EPA":
            z = F.normalize(self.epa_projector(out["eeg_emb_vector"]).float(), dim=-1)
            logits = z @ self.epa_prototypes.float().T / self.epa_temperature
            out["epa_projection"] = z
            out["epa_logits"] = logits
            out["loss_epa"] = weighted_epa_ce(logits, out["emotion_targets"],
                                              batch["text uid"], self.train_uid_counts or {})
            out["loss_total"] = out["loss_total"] + self.epa_weight * out["loss_epa"]
        return out

    def log_auxiliary(self, out, phase):
        super().log_auxiliary(out, phase)
        if self.arm == "B1_EPA":
            self.log(f"{phase}/loss_epa", out["loss_epa"],
                     batch_size=len(out["emotion_targets"]), sync_dist=True)
