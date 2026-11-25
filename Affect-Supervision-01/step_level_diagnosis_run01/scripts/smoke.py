"""Single real-batch forward/backward preflight; never calls optimizer.step()."""

import json
import os
from pathlib import Path

import lightning as L
import torch

from experiment_assets import build_arm, read_v3, verify_assets
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8-sig"))
checks = verify_assets(config, include_large_data=True)
os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
L.seed_everything(42, workers=True)
package = read_v3(config)
model = build_arm(config, package, "B0_Baseline", 42).cuda()
model.setup("fit")
model.train()
model.text_model.eval()
data = Task1DataModule(data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
                       expected_manifest=config["assets"]["e1_manifest"],
                       expected_sha256=config["assets"]["e1_manifest_sha256"],
                       audit_dir=ROOT / "reports" / "smoke", seed=42,
                       bsz_train=72, bsz_val=24, bsz_test=24, num_workers=4)
data.setup("fit")
loader = data.train_dataloader()
batch = next(iter(loader))
batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.shared_forward(batch)
out["loss_total"].backward()
if not torch.isfinite(out["loss_total"]):
    raise RuntimeError("Nonfinite forward")
if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
    raise RuntimeError("Nonfinite backward")
if any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad):
    raise RuntimeError("Parameters changed without optimizer.step")
record = {"status": "PASS", "dataloader_batches": len(loader),
          "batch_size": int(batch["eeg"].shape[0]), "text_model_eval": not model.text_model.training,
          "optimizer_steps": 0, "source_hashes": checks,
          "losses": {key: float(out[key].detach().float().cpu()) for key in
                     ("loss_lm", "loss_clip", "loss_vq", "loss_disc_emotion", "loss_total")}}
(ROOT / "reports" / "smoke.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2), flush=True)
