"""Synthetic tensor tests only. No real EEG, optimization steps or downloads."""
import argparse
import inspect
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.emotion_head import EmotionHead, emotion_loss
from model.vq import TokenVQ


def tensor_only_imports():
    """Opt-in import adapters for a machine without Lightning/TorchMetrics.
    They support model construction only. No Trainer or metric is simulated.
    """
    class Module(torch.nn.Module):
        @property
        def device(self):
            return next(self.parameters()).device

        def save_hyperparameters(self, values=None, **kwargs):
            if values is None:
                frame = inspect.currentframe().f_back
                values = {k: frame.f_locals[k] for k in inspect.signature(
                    frame.f_locals["__class__"].__init__).parameters if k != "self"}
            if not hasattr(self, "hparams"):
                self.hparams = types.SimpleNamespace()
            vars(self.hparams).update(values)

    def unavailable(*args, **kwargs):
        raise AssertionError("Framework metrics are outside the tensor-only test")

    imports = {}
    for name, symbols in {
        "lightning": {"LightningModule": Module, "LightningDataModule": object},
        "lightning.pytorch.utilities": {"rank_zero_only": unavailable},
        "torchmetrics.functional.classification": dict.fromkeys(["multiclass_accuracy", "binary_accuracy"], unavailable),
        "torchmetrics.functional.text": dict.fromkeys(["bleu_score", "rouge_score", "word_error_rate"], unavailable),
    }.items():
        imports[name] = types.ModuleType(name)
        vars(imports[name]).update(symbols)
    return patch.dict(sys.modules, imports)


def test_vq():
    vq = TokenVQ(5, 3, 7)
    x = torch.randn(2, 4, 5, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    out = vq(x, mask)
    assert out["quantized"].shape == (2, 4, 3)
    assert out["code_usage_histogram"].sum() == mask.sum()
    assert (out["code_ids"][~mask] == -1).all()
    torch.testing.assert_close(out["quantized"][mask], vq.codebook(out["code_ids"][mask]))
    changed = x.detach().masked_fill(~mask.unsqueeze(-1), 1234)
    for key, value in vq(changed, mask).items():
        torch.testing.assert_close(out[key], value)
    head = EmotionHead(5, 4, 3)
    torch.testing.assert_close(head(x, mask), head(changed, mask))
    z = vq.projection(x[mask])
    q = vq.codebook(out["code_ids"][mask])
    torch.testing.assert_close(out["loss_vq_codebook"], F.mse_loss(q, z.detach()))
    torch.testing.assert_close(out["loss_vq_commitment"], 0.25 * F.mse_loss(z, q.detach()))
    out["quantized"].sum().backward(retain_graph=True)
    assert x.grad[mask].abs().sum() > 0 and not x.grad[~mask].any() and vq.codebook.weight.grad is None
    out["loss_vq"].backward()
    assert vq.codebook.weight.grad.abs().sum() > 0 and not x.grad[~mask].any()
    empty = vq(x, torch.zeros_like(mask))
    assert all(torch.isfinite(v).all() for v in empty.values())
    assert empty["loss_vq"] == empty["code_usage_histogram"].sum() == 0
    logits = torch.randn(2, 3, requires_grad=True)
    torch.testing.assert_close(emotion_loss(logits, torch.tensor([2, -1])), -logits.log_softmax(-1)[0, 2])
    emotion_loss(logits, torch.tensor([-1, -1])).backward()
    assert not logits.grad.any()
    print("PASS synthetic VQ, padding, STE, codebook gradients and label masking", flush=True)


def test_variants(args):
    from model.glim import GLIM
    from model.glim_multitask import GLIMMultitask, VARIANTS
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, local_files_only=True)
    lm = T5ForConditionalGeneration.from_pretrained(args.text_model_path, local_files_only=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).requires_grad_(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]

    def prepare(model):
        model.to(device)
        model.tokenizer, model.text_model = tokenizer, lm
        return model.eval()

    batch = {"eeg": torch.randn(2, 1280, 128, device=device),
             "mask": torch.ones(2, 1280, device=device, dtype=torch.int8),
             "prompt": [("<NR>", "<NR>"), ("ZuCo1", "ZuCo1"), ("ZAB", "ZDM")],
             "input text": ["To English: synthetic A", "To English: synthetic B"],
             "target text": ["synthetic A", "synthetic B"], "raw input text": ["A", "B"],
             "all target texts": [("A", "B")], "raw task key": ["task1", "task2"],
             "sentiment label": ["positive", "nan"], "relation label": ["nan", "nan"]}
    batch["mask"][0, 500:] = 0
    original = GLIM(**config)
    original.load_state_dict(ckpt["state_dict"], strict=True)
    prepare(original)
    # This single test scope switches from no_grad/eval to training; do not reuse
    # casts cached under no_grad for the later differentiable forward.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                        cache_enabled=False):
        with torch.no_grad():
            reference = original.shared_forward(batch)
        for variant in ("A3-full", "A3-no-attribute", "A3-continuous-only"):
            revised = prepare(GLIMMultitask.from_glim_checkpoint(args.checkpoint, variant))
            with torch.no_grad():
                output = revised(batch)
                if variant == "A3-full":
                    assert output["eeg_embeds"].shape == (2, 97, 1024)
                    assert output["memory_mask"][:,0].tolist() == [True, False]
                    for other in ("task2", "task3"):
                        changed = revised({**batch, "raw task key": ["task1", other],
                                           "sentiment label": ["negative","positive"]})
                        assert changed["emotion_targets"].tolist() == [0,-100]
                        torch.testing.assert_close(changed["logits_lm"], output["logits_lm"], rtol=0, atol=0)
                    generated = revised.generation_step(output)["gen_ids"]
                    _, texts = revised.predict(batch["eeg"],batch["mask"],batch["prompt"],
                                               generate=True,raw_tasks=batch["raw task key"])
                    assert texts == tokenizer.batch_decode(generated, skip_special_tokens=True)
                else:
                    torch.testing.assert_close(output["logits_lm"], reference["logits_lm"], rtol=0, atol=0)
                    assert ("disc_logits" in output) == (variant == "A3-no-attribute")
            if variant == "A3-full":
                logits = torch.randn(2,3,device=device,requires_grad=True)
                token = revised.attribute(logits)
                token.sum().backward()
                assert logits.grad is None
                assert all(p.grad is not None and p.grad.any() for p in revised.attribute.parameters())
            print(f"PASS {variant} synthetic forward, conditioning and ablation checks",flush=True)
            del revised
        for variant in ("baseline", "continuous", "discrete", "dual"):
            model = GLIMMultitask.from_glim_checkpoint(args.checkpoint, variant)
            assert not model.load_report["unexpected_keys"]
            for key, weight in ckpt["state_dict"].items():
                torch.testing.assert_close(model.state_dict()[key], weight, rtol=0, atol=0)
            prepare(model)
            with torch.no_grad():
                out = model(batch)
                for key, value in reference.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(out[key], value, atol=0, rtol=0)
                    else:
                        assert out[key] == value
                if variant == "baseline":
                    assert out.keys() == reference.keys() and model.state_dict().keys() == original.state_dict().keys()
                expected = .5 * out["loss_clip"] + .5 * out["loss_lm"] + model.ε * out["loss_commitment"]
                heads = []
                for branch, enabled in (("cont", variant in ("continuous", "dual")),
                                        ("disc", variant in ("discrete", "dual"))):
                    assert (branch + "_logits" in out) == enabled
                    if enabled:
                        logits = out[branch + "_logits"]
                        assert logits.shape == (2, 3)
                        heads.append(F.cross_entropy(logits[:1].float(), torch.tensor([2], device=device)))
                if heads:
                    expected += .2 * torch.stack(heads).mean()
                if "loss_vq" in out:
                    expected += out["loss_vq_codebook"] + out["loss_vq_commitment"]
                    assert out["code_usage_histogram"].sum() == 2 * 96
                assert all(torch.isfinite(v).all() for k, v in out.items() if k.startswith("loss"))
                # Exercise the actual training objective; retrieval and logging are orthogonal.
                with patch.object(model, "shared_forward", return_value=out), patch.object(
                    model, "cal_retrieval_metrics", return_value={}), patch.object(
                    model, "log_dict", create=True), patch.object(type(model), "current_epoch", 1, create=True):
                    torch.testing.assert_close(model.training_step(batch, 1), expected)
            if variant == "dual":
                with torch.no_grad():
                    altered = batch["eeg"].masked_fill(~batch["mask"].bool().unsqueeze(-1), 19)
                    predicted = model.predict_emotion(altered, batch["mask"], batch["prompt"])
                    for key in ("cont_logits", "disc_logits", "code_ids", "code_usage_histogram"):
                        torch.testing.assert_close(predicted[key], out[key], atol=0, rtol=0)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "synthetic.ckpt"
                    params = vars(model.hparams) if args.tensor_only else dict(model.hparams)
                    saved = dict(hyper_parameters=params, state_dict=model.state_dict())
                    model.on_save_checkpoint(saved)
                    assert not any(k.startswith("text_model.") for k in saved["state_dict"])
                    torch.save(saved, path)
                    restored = GLIMMultitask.from_glim_checkpoint(path)
                    assert restored.variant == variant and not restored.load_report["missing_keys"]
                    for k, v in restored.state_dict().items():
                        torch.testing.assert_close(v, saved["state_dict"][k].cpu(), atol=0, rtol=0)
                model.train()
                training = model(batch)
                training["loss_total"].backward()
                for module in (model.eeg_encoder, model.cont_head, model.vq.codebook, model.disc_head):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    assert grads and all(g.isfinite().all() for g in grads) and any(g.any() for g in grads), type(module)
                assert all(p.grad is None for p in lm.parameters())
                print("PASS synthetic dual: raw padding invariance, checkpoint round trip, train-mode gradients", flush=True)
            print(f"PASS synthetic {variant}: checkpoint, paired forward, logits and training objective", flush=True)


if __name__ == "__main__":
    from contextlib import nullcontext
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text-model-path", required=True)
    p.add_argument("--tensor-only", action="store_true", help="Use framework import adapters (no Trainer coverage)")
    args = p.parse_args()
    torch.manual_seed(42)
    test_vq()
    with tensor_only_imports() if args.tensor_only else nullcontext():
        test_variants(args)
