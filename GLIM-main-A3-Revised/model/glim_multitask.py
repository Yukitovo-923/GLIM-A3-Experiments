import logging

import torch
import torch.distributed as dist

from .emotion_head import EmotionHead, emotion_loss
from .glim import GLIM
from .vq import TokenVQ, usage_stats


REVISED = ("A3-full", "A3-no-attribute", "A3-continuous-only")
VARIANTS = ("baseline", "continuous", "discrete", "dual") + REVISED


class GLIMMultitask(GLIM):
    def __init__(self, variant="baseline", codebook_size=128, code_dim=256, beta=0.25,
                 emotion_hidden_dim=128, emotion_weight=0.2, vq_weight=1.0, disc_emotion_weight=0.2, cont_emotion_weight=0.0,
                 lm_loss_weight=0.5, attribute_dim=64, temperature=1.0, gate_init=0.1, detach_probs=True, **glim_config):
        super().__init__(**glim_config)
        if variant not in VARIANTS:
            raise ValueError(f"Expected one of {VARIANTS}, got {variant!r}")
        self.save_hyperparameters(dict(variant=variant, codebook_size=codebook_size,
                                       code_dim=code_dim, beta=beta, emotion_hidden_dim=emotion_hidden_dim,
                                       emotion_weight=emotion_weight, vq_weight=vq_weight,
                                       disc_emotion_weight=disc_emotion_weight, cont_emotion_weight=cont_emotion_weight,
                                       lm_loss_weight=lm_loss_weight, attribute_dim=attribute_dim, temperature=temperature, gate_init=gate_init,
                                       detach_probs=detach_probs))
        self.variant = variant
        dim = self.eeg_encoder.out_dim
        classes = len(self.sentiment_labels)
        if variant in ("continuous", "dual") or (variant in REVISED[:2] and cont_emotion_weight > 0):
            self.cont_head = EmotionHead(dim, emotion_hidden_dim, classes)
        if variant in ("discrete", "dual", *REVISED[:2]):
            self.vq = TokenVQ(dim, code_dim, codebook_size, beta)
            self.disc_head = EmotionHead(code_dim, emotion_hidden_dim, classes)
        if variant == "A3-full":
            from .emotion_head import EmotionAttribute
            self.attribute = EmotionAttribute(attribute_dim, self.embed_dim, temperature, gate_init, detach_probs)
        self._epoch_counts = {}

    def emotion_outputs(self, hidden, eeg_mask):
        # Each learned query attends to all unpadded EEG frames. Query indices have
        # no temporal correspondence to the input mask. Every query is valid
        # when the sample contains at least one real frame.
        mask = eeg_mask.bool().any(dim=1, keepdim=True).expand(hidden.shape[:2])
        out = {}
        if hasattr(self, "cont_head"):
            out["cont_logits"] = self.cont_head(hidden, mask)
            out["cont_predictions"] = out["cont_logits"].argmax(-1)
        if hasattr(self, "vq"):
            vq = self.vq(hidden, mask)
            out["disc_logits"] = self.disc_head(vq.pop("quantized"), mask)
            out["disc_predictions"] = out["disc_logits"].argmax(-1)
            out.update(vq)
        return out

    def condition_memory(self, hidden, memory, eeg_mask, tasks, datasets, base_mask=None):
        extra = self.emotion_outputs(hidden, eeg_mask)
        sr = torch.tensor([t == "task1" and d == "ZuCo1" for t, d in zip(tasks, datasets)],
                          device=memory.device) & eeg_mask.bool().any(1)
        if hasattr(self, "attribute"):
            token = self.attribute(extra["disc_logits"]).to(memory.dtype)
            base = torch.ones_like(memory[..., 0], dtype=torch.bool) if base_mask is None else base_mask.bool()
            extra["memory_mask"] = torch.cat((sr[:, None], base), dim=1)
            memory = torch.cat((token[:, None], memory), dim=1)
        return memory, extra

    def shared_forward(self, batch):
        if self.variant in ("baseline", "A3-continuous-only"):
            return super().shared_forward(batch)
        if self.variant in REVISED:
            def build(hidden, memory, text_mask):
                base_mask = text_mask if self.use_y_mask and self.training else None
                return self.condition_memory(hidden, memory, batch["mask"], batch["raw task key"],
                                             batch["prompt"][1], base_mask)
            out = super().shared_forward(batch, memory_builder=build)
        else:
            out, hidden = super().shared_forward(batch, return_eeg_hiddens=True)
            out.update(self.emotion_outputs(hidden, batch["mask"]))
        labels = out["sentiment_ids"].masked_fill(~batch["mask"].bool().any(1), -1)
        if self.variant in REVISED:
            sr = (out["raw_task_ids"] == self.raw_task_keys.index("task1")) & (out["prompt_ids"][:, 1] == self.prompt_keys["dataset"].index("ZuCo1"))
            labels = labels.masked_fill(~sr | labels.lt(0), -100)
        emotion_losses = []
        for branch in ("cont", "disc"):
            if branch + "_logits" in out:
                loss = emotion_loss(out[branch + "_logits"], labels)
                out[f"loss_{branch}_emotion"] = loss
                emotion_losses.append(loss)
        out["emotion_targets"] = labels
        out["loss_glim_commitment"] = out["loss_commitment"]
        lm_weight = self.hparams.lm_loss_weight if self.variant in REVISED else 1 - self.λ
        out["loss_glim"] = (self.λ * out["loss_clip"] + lm_weight * out["loss_lm"]
                            + self.ε * out["loss_glim_commitment"])
        if self.variant in REVISED:
            total = out["loss_glim"] + self.hparams.disc_emotion_weight * out["loss_disc_emotion"]
            if "loss_cont_emotion" in out:
                total = total + self.hparams.cont_emotion_weight * out["loss_cont_emotion"]
        else:
            total = out["loss_glim"] + self.hparams.emotion_weight * torch.stack(emotion_losses).mean()
        if "loss_vq" in out:
            total = total + self.hparams.vq_weight * out["loss_vq"]
        out["loss_total"] = total
        return out

    def forward(self, batch):
        return self.shared_forward(batch)

    @torch.no_grad()
    def predict_emotion(self, eeg, eeg_mask, prompts):
        """Text-free inference for the enabled emotion heads; call eval() first."""
        if self.variant in ("baseline", "A3-continuous-only"):
            return {}
        ids = self.p_embedder.encode(prompts, device=eeg.device)
        hidden, _ = self.eeg_encoder(eeg, eeg_mask, self.p_embedder(ids, self.eval_pembed))
        return self.emotion_outputs(hidden, eeg_mask)

    def log_auxiliary(self, out, phase):
        labels = out["emotion_targets"]
        losses = {f"{phase}/{key}": value for key, value in out.items()
                  if key.startswith(("loss_glim", "loss_vq", "loss_cont", "loss_disc", "loss_total"))}
        self.log_dict(losses, batch_size=len(labels), sync_dist=True)
        valid = labels.ge(0)
        counts = {"labeled": valid.sum()}
        for branch in ("cont", "disc"):
            if branch + "_predictions" in out:
                counts[branch] = torch.bincount(3 * labels[valid].long() + out[branch + "_predictions"][valid],
                                                 minlength=9).reshape(3, 3)
        if "code_usage_histogram" in out:
            counts["hist"] = out["code_usage_histogram"]
        totals = self._epoch_counts.setdefault(phase, {})
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value.detach()

    def _log_epoch_counts(self, phase):
        counts = self._epoch_counts.pop(phase, {})
        if not counts:
            return
        if dist.is_initialized():
            for value in counts.values():
                dist.all_reduce(value)
        logs = {f"{phase}/emotion_labeled": counts["labeled"].float()}
        for branch in ("cont", "disc"):
            if branch in counts and counts["labeled"].item():
                cm = counts[branch].float()
                tp = cm.diag()
                precision = tp / cm.sum(0).clamp_min(1)
                recall = tp / cm.sum(1).clamp_min(1)
                f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
                prefix = f"{phase}/{branch}"
                logs.update({prefix+"_accuracy": tp.sum()/cm.sum(), prefix+"_macro_f1": f1.mean(),
                             prefix+"_balanced_accuracy": recall[cm.sum(1)>0].mean()})
                for i, label in enumerate(self.sentiment_labels):
                    for metric, val in (("precision",precision[i]),("recall",recall[i]),("f1",f1[i])):
                        logs[f"{prefix}/{label}/{metric}"] = val
                    for j, pred in enumerate(self.sentiment_labels):
                        logs[f"{prefix}/confusion/{label}/{pred}"] = cm[i,j]
        if "hist" in counts:
            hist = counts["hist"]
            logs[f"{phase}/codebook_perplexity"], logs[f"{phase}/active_code_ratio"] = usage_stats(hist)
            logs.update({f"{phase}/code_usage/{i}": value.float() for i, value in enumerate(hist)})
        self.log_dict(logs, sync_dist=False)

    def on_train_epoch_end(self):
        self._log_epoch_counts("train")

    def on_validation_epoch_end(self):
        self._log_epoch_counts("val")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self._log_epoch_counts("test")
        super().on_test_epoch_end()

    @classmethod
    def from_glim_checkpoint(cls, path, variant=None, **overrides):
        """Initialize from a trusted original or experiment checkpoint before setup()."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(ckpt["hyper_parameters"])
        saved_variant = config.get("variant")
        if saved_variant and variant and saved_variant != variant:
            raise ValueError(f"Checkpoint variant {saved_variant} differs from {variant}")
        model = cls(**{**config, **overrides, "variant": variant or saved_variant or "baseline"})
        state = ckpt["state_dict"]
        expected = set(model.state_dict())
        missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
        model.load_report = {"missing_keys": missing, "unexpected_keys": unexpected}
        logging.getLogger(__name__).warning("GLIM checkpoint %s: %s", path, model.load_report)
        allowed = () if saved_variant else ("cont_head.", "disc_head.", "vq.", "attribute.")
        if unexpected or any(not k.startswith(allowed) for k in missing):
            raise RuntimeError(f"GLIM checkpoint keys do not match: {model.load_report}")
        model.load_state_dict(state, strict=not missing)  # shape mismatches always raise
        return model
