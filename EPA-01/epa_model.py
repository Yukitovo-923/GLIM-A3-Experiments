"""Independent EPA branch on GLIM's EEG-side CLIP query vector."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.glim_multitask import GLIMMultitask

CLASSES = ("negative", "neutral", "positive")
ARMS = ("E0", "E1", "E2")


def masked_sentence_ce(logits, labels, uid, uid_counts=None):
    """Ignore unlabeled/invalid trials; inverse train-trial counts balance UIDs."""
    labels = labels.long()
    valid = labels.ge(0)
    if uid_counts is None:
        weights = torch.ones_like(labels, dtype=torch.float32)
    else:
        weights = torch.tensor([1.0 / uid_counts[int(u)] if int(u) in uid_counts else 1.0
                                for u in uid.detach().cpu().tolist()],
                               dtype=torch.float32, device=logits.device)
    weights = weights * valid.float()
    per = F.cross_entropy(logits.float(), labels.masked_fill(~valid, -100),
                          ignore_index=-100, reduction="none")
    return (per * weights).sum() / weights.sum().clamp_min(1e-12)


def classification(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    for target, predicted in zip(y, pred):
        cm[int(target), int(predicted)] += 1
    support = cm.sum(1)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * tp, support + cm.sum(0), out=np.zeros(3),
                   where=(support + cm.sum(0)) > 0)
    return {"n": int(len(y)), "support": support.tolist(), "predicted": cm.sum(0).tolist(),
            "accuracy": float(tp.sum() / max(1, cm.sum())),
            "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support) else 0.0,
            "precision": precision.tolist(), "recall": recall.tolist(),
            "class_f1": f1.tolist(), "confusion": cm.tolist()}


class EPAExperiment(GLIMMultitask):
    def __init__(self, *, arm, prototypes_path, prototype_sha256, epa_tau=0.07,
                 emotion_aux_weight=0.2, metrics_dir=None, **glim_config):
        if arm not in ARMS or epa_tau <= 0 or emotion_aux_weight < 0:
            raise ValueError("Invalid EPA arm, temperature or weight")
        if glim_config.get("variant") != "A3-no-attribute":
            raise ValueError("All EPA arms must use A3-no-attribute generator")
        if glim_config.get("disc_emotion_weight") != 0.0 or glim_config.get("cont_emotion_weight") != 0.0:
            raise ValueError("Base emotion weights must be zero; EPA controls the sole emotion loss")
        super().__init__(**glim_config)
        if hasattr(self, "attribute"):
            raise RuntimeError("Attribute token would confound EPA arms")
        vectors = torch.as_tensor(np.load(prototypes_path, allow_pickle=False), dtype=torch.float32)
        if vectors.shape != (3, self.embed_dim) or not torch.isfinite(vectors).all():
            raise ValueError("Expected finite [3,1024] text prototypes")
        if not torch.allclose(vectors.norm(dim=1), torch.ones(3), atol=1e-5):
            raise ValueError("Text prototypes must be unit vectors")
        self.register_buffer("epa_prototypes", vectors.clone(), persistent=True)
        self.epa_projector = nn.Linear(self.embed_dim, self.embed_dim)
        self.arm = arm
        self.epa_tau = float(epa_tau)
        self.emotion_aux_weight = float(emotion_aux_weight)
        self.prototype_sha256 = prototype_sha256
        self.uid_trial_counts = None
        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else None
        self._evaluation_rows = defaultdict(list)
        self.save_hyperparameters(dict(epa_arm=arm, epa_tau=epa_tau,
                                       emotion_aux_weight=emotion_aux_weight,
                                       prototype_sha256=prototype_sha256,
                                       epa_target_space="unprefixed_flan_t5_encoder_mean_external_auxiliary"))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Restore every EPA/GLIM key; frozen T5 is supplied by the verified local model."""
        if strict and hasattr(self, "text_model") and not any(
                key.startswith("text_model.") for key in state_dict):
            expected = set(self.state_dict())
            supplied = set(state_dict)
            missing = expected - supplied
            unexpected = supplied - expected
            bad_missing = sorted(key for key in missing if not key.startswith("text_model."))
            if unexpected or not missing or bad_missing:
                raise RuntimeError(f"EPA resume has non-T5 missing or unexpected keys: "
                                   f"missing={bad_missing}, "
                                   f"unexpected={sorted(unexpected)}")
            if any(parameter.requires_grad for parameter in self.text_model.parameters()):
                raise RuntimeError("Resume requires a frozen local T5 model")
            report = super().load_state_dict(state_dict, strict=False, assign=assign)
            if set(report.missing_keys) != missing or report.unexpected_keys:
                raise RuntimeError("EPA resume key audit changed during load")
            self.resume_load_report = {
                "restored_non_t5_keys": len(supplied),
                "frozen_local_t5_keys": len(missing),
                "missing_non_t5_keys": [], "unexpected_keys": []}
            print("EPA_CHECKPOINT_LOAD " + json.dumps(self.resume_load_report), flush=True)
            return report
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_train_uid_counts(self, counts):
        self.uid_trial_counts = {int(k): int(v) for k, v in counts.items()}
        if not self.uid_trial_counts or min(self.uid_trial_counts.values()) <= 0:
            raise ValueError("Missing train sentence trial counts")

    def on_fit_start(self):
        super().on_fit_start()
        counts = getattr(self.trainer.datamodule, "uid_trial_counts", None)
        if counts is None:
            raise RuntimeError("EPA requires audited train sentence trial counts")
        self.set_train_uid_counts(counts)

    def shared_forward(self, batch):
        out = super().shared_forward(batch)
        # EEG-only q_x embedding: q_y and T5 do not appear in this branch.
        eeg_vector = out["eeg_emb_vector"]
        z = F.normalize(self.epa_projector(eeg_vector).float(), dim=-1)
        logits = z @ self.epa_prototypes.float().T / self.epa_tau
        labels = out["emotion_targets"]
        uid = batch["text uid"]
        counts = self.uid_trial_counts if self.training else None
        out["epa_projection"] = z
        out["epa_logits"] = logits
        out["epa_predictions"] = logits.argmax(-1)
        out["loss_epa"] = masked_sentence_ce(logits, labels, uid, counts)
        out["loss_disc_emotion_balanced"] = masked_sentence_ce(out["disc_logits"], labels, uid, counts)
        # The three arms have byte-identical GLIM/VQ paths and initialization.
        non_emotion = out["loss_glim"] + self.hparams.vq_weight * out["loss_vq"]
        emotion = {"E0": non_emotion.new_zeros(()),
                   "E1": out["loss_disc_emotion_balanced"],
                   "E2": out["loss_epa"]}[self.arm]
        out["loss_total"] = non_emotion + self.emotion_aux_weight * emotion
        out["epa_input_shape"] = tuple(eeg_vector.shape)
        out["epa_projection_shape"] = tuple(z.shape)
        out["epa_subjects"] = tuple(batch["prompt"][2])
        out["epa_uids"] = uid
        return out

    def log_auxiliary(self, out, phase):
        super().log_auxiliary(out, phase)
        n = out["epa_logits"].shape[0]
        self.log(f"{phase}/loss_epa", out["loss_epa"], batch_size=n, sync_dist=True)
        self.log(f"{phase}/loss_disc_emotion_balanced", out["loss_disc_emotion_balanced"],
                 batch_size=n, sync_dist=True)
        if phase not in ("val", "test"):
            return
        y = out["emotion_targets"].detach().cpu().numpy()
        epa_probs = out["epa_logits"].float().softmax(-1).detach().cpu().numpy()
        head_probs = out["disc_logits"].float().softmax(-1).detach().cpu().numpy()
        uids = out["epa_uids"].detach().cpu().numpy()
        for i in range(n):
            if int(y[i]) >= 0:
                self._evaluation_rows[phase].append((int(uids[i]), str(out["epa_subjects"][i]),
                                                     int(y[i]), epa_probs[i], head_probs[i]))

    def _save_grouped_metrics(self, phase):
        rows = self._evaluation_rows.pop(phase, [])
        if not rows:
            return
        result = {"arm": self.arm, "phase": phase, "epoch": int(self.current_epoch),
                  "class_order": list(CLASSES), "heads": {}}
        uid = np.array([r[0] for r in rows], dtype=int)
        subjects = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows], dtype=int)
        for name, offset in (("epa", 3), ("original_emotion_head", 4)):
            probabilities = np.stack([r[offset] for r in rows])
            target = y
            result["heads"][name] = {"trial_level": classification(target, probabilities.argmax(1)),
                                     "by_subject": {}}
            grouped = defaultdict(list)
            for i, sentence_uid in enumerate(uid):
                grouped[int(sentence_uid)].append(i)
            gy, gp = [], []
            for indices in grouped.values():
                if len(set(target[indices])) != 1:
                    raise ValueError("Conflicting labels for one sentence UID")
                gy.append(int(target[indices[0]]))
                gp.append(probabilities[indices].mean(0))
            result["heads"][name]["sentence_level"] = classification(np.array(gy),
                                                                         np.stack(gp).argmax(1))
            for subject in np.unique(subjects):
                indices = np.flatnonzero(subjects == subject)
                sy, sp = [], []
                for sentence_uid in np.unique(uid[indices]):
                    j = indices[uid[indices] == sentence_uid]
                    sy.append(int(target[j[0]]))
                    sp.append(probabilities[j].mean(0))
                result["heads"][name]["by_subject"][str(subject)] = {
                    "trial_level": classification(target[indices], probabilities[indices].argmax(1)),
                    "sentence_level": classification(np.array(sy), np.stack(sp).argmax(1))}
            selected = (name == ("epa" if self.arm == "E2" else "original_emotion_head"))
            if selected:
                self.log(f"{phase}/sentiment_trial_macro_f1",
                         result["heads"][name]["trial_level"]["macro_f1"], sync_dist=False)
                self.log(f"{phase}/sentiment_sentence_macro_f1",
                         result["heads"][name]["sentence_level"]["macro_f1"], sync_dist=False)
        if self.metrics_dir is not None and self.trainer.is_global_zero:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            path = self.metrics_dir / f"{phase}_epoch={self.current_epoch:03d}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._save_grouped_metrics("val")

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._save_grouped_metrics("test")
