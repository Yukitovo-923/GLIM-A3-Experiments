"""No-training EPA resume smoke: verify frozen T5 omission and optimizer restore."""
import json
import os

import torch

from epa_assets import load_model_for_arm, load_common, hash_file

EPA = "/root/autodl-tmp/EPA-01"
common = load_common(EPA + "/common.json")
os.environ["GLIM_TEXT_MODEL_PATH"] = common["assets"]["text_model"]
source = EPA + "/runs/E0-01/checkpoints/epoch=1-step=3976.ckpt"
model = load_model_for_arm(EPA + "/common.json", EPA + "/initialization/common_init.pt", "E0").cuda()
model.setup("fit")
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
if checkpoint["epoch"] != 1 or checkpoint["global_step"] != 3976:
    raise RuntimeError("Wrong resume source epoch/step")
model.load_state_dict(checkpoint["state_dict"], strict=True)
optimizer = model.configure_optimizers()
optimizer.load_state_dict(checkpoint["optimizer_states"][0])
if not optimizer.state or not model.resume_load_report["frozen_local_t5_keys"]:
    raise RuntimeError("Optimizer or frozen T5 restore failed")
if any(parameter.requires_grad for parameter in model.text_model.parameters()):
    raise RuntimeError("T5 was unexpectedly trainable")
print(json.dumps({"status": "PASS", "resume_checkpoint_sha256": hash_file(source),
                  "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
                  "restored_non_t5_keys": model.resume_load_report["restored_non_t5_keys"],
                  "frozen_local_t5_keys": model.resume_load_report["frozen_local_t5_keys"],
                  "optimizer_states": len(optimizer.state),
                  "text_model_sha256": common["assets"]["text_model_weight_sha256"]}, indent=2))
