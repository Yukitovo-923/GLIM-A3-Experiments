"""Read-only VQ geometry diagnostic on a small real ZuCo validation batch."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datamodule import GLIMDataModule
from model.glim_multitask import GLIMMultitask


def summarize(model, batches):
    model.eval()
    all_z = []
    all_h = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches:
            eeg = batch["eeg"].cuda()
            mask = batch["mask"].cuda()
            prompt_ids = model.p_embedder.encode(batch["prompt"], device=eeg.device)
            hidden, _ = model.eeg_encoder(eeg, mask, model.p_embedder(prompt_ids, model.eval_pembed))
            valid = mask.bool().any(1, keepdim=True).expand(hidden.shape[:2])
            z = model.vq.projection(hidden[valid]).float()
            all_z.append(z)
            all_h.append(hidden[valid].float())
    z = torch.cat(all_z)
    h = torch.cat(all_h)
    codes = model.vq.codebook.weight.float()
    distances = codes.square().sum(-1)[None, :] - 2 * z @ codes.T
    ids = distances.argmin(-1)
    hist = torch.bincount(ids, minlength=codes.shape[0])
    normed_ids = (torch.nn.functional.normalize(z, dim=-1) @
                  torch.nn.functional.normalize(codes, dim=-1).T).argmax(-1)
    normed_hist = torch.bincount(normed_ids, minlength=codes.shape[0])
    top2 = distances.topk(2, largest=False).values
    code_norm = codes.norm(dim=-1)
    code94 = 94
    return {
        "valid_tokens": int(z.shape[0]),
        "hidden_norm_mean": h.norm(dim=-1).mean().item(),
        "projected_norm_mean": z.norm(dim=-1).mean().item(),
        "projected_norm_std": z.norm(dim=-1).std().item(),
        "projected_feature_std_mean": z.std(dim=0).mean().item(),
        "code_norm_min": code_norm.min().item(),
        "code_norm_median": code_norm.median().item(),
        "code_norm_max": code_norm.max().item(),
        "code94_norm": code_norm[code94].item(),
        "code94_norm_rank_smallest_first": int((code_norm < code_norm[code94]).sum().item()) + 1,
        "euclidean_active_codes": int(hist.gt(0).sum().item()),
        "euclidean_top_codes": [[int(i), int(hist[i])] for i in hist.argsort(descending=True)[:8]],
        "cosine_active_codes": int(normed_hist.gt(0).sum().item()),
        "cosine_top_codes": [[int(i), int(normed_hist[i])] for i in normed_hist.argsort(descending=True)[:8]],
        "euclidean_mean_second_best_margin": (top2[:, 1] - top2[:, 0]).mean().item(),
        "projection_weight_norm": model.vq.projection.weight.float().norm().item(),
        "projection_bias_norm": model.vq.projection.bias.float().norm().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--epoch4-checkpoint", type=Path, required=True)
    parser.add_argument("--epoch49-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = config.pop("variant")
    model = GLIMMultitask.from_glim_checkpoint(args.original_checkpoint, variant, **config)
    initial_codes = model.vq.codebook.weight.detach().cpu().clone()
    model.cuda().eval()
    dm = GLIMDataModule(data_path=str(args.data_path), bsz_train=model.bsz_train,
                        bsz_val=model.bsz_val, bsz_test=model.bsz_retrieval, num_workers=0)
    dm.label_reference = args.labels
    dm.manifest_path = args.output.with_name("vq_diagnostic_manifest.csv")
    dm.setup("fit")
    iterator = iter(dm.val_dataloader())
    batches = [next(iterator), next(iterator)]
    result = {"kind": "REAL_ZUCO_VQ_GEOMETRY_DIAGNOSTIC",
              "batch_count": len(batches), "batch_sizes": [len(b["sentiment label"]) for b in batches],
              "states": {"initial_seed42_reconstruction": summarize(model, batches)}}
    previous_codes = initial_codes
    for name, path, expected_epoch in (("epoch4", args.epoch4_checkpoint, 4),
                                       ("epoch49", args.epoch49_checkpoint, 49)):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["epoch"] != expected_epoch:
            raise RuntimeError(f"Unexpected checkpoint epoch at {path}")
        codes = saved["state_dict"]["vq.codebook.weight"].detach().cpu()
        changed = (codes - previous_codes).norm(dim=-1)
        model.load_state_dict(saved["state_dict"], strict=True)
        entry = summarize(model, batches)
        entry["code_rows_changed_since_previous_state_gt_1e-6"] = int(changed.gt(1e-6).sum().item())
        entry["code94_change_since_previous_state"] = changed[94].item()
        entry["largest_changed_code_rows"] = [[int(i), changed[i].item()]
                                               for i in changed.argsort(descending=True)[:8]]
        result["states"][name] = entry
        previous_codes = codes
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
