import torch
from torch import nn


class TokenVQ(nn.Module):
    def __init__(self, input_dim, code_dim=256, codebook_size=128, beta=0.25):
        super().__init__()
        if min(input_dim, code_dim, codebook_size) <= 0 or beta < 0:
            raise ValueError("VQ dimensions must be positive and beta must be nonnegative")
        self.projection = nn.Linear(input_dim, code_dim)
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.beta = beta

    def forward(self, hidden, mask):
        if mask.shape != hidden.shape[:2]:
            raise ValueError("Expected a token mask with shape (B, L)")
        mask = mask.bool()
        # Select before projection so padding contributes neither gradients nor counts.
        z = self.projection(hidden[mask])
        with torch.autocast(device_type=z.device.type, enabled=False):
            features, codes = z.float(), self.codebook.weight.float()
            distances = codes.square().sum(-1) - 2 * features @ codes.T
            ids = distances.argmin(-1)
            q = self.codebook(ids).float()
            denominator = max(features.numel(), 1)
            code_loss = (q - features.detach()).square().sum() / denominator
            commit_loss = self.beta * (features - q.detach()).square().sum() / denominator
        straight_through = q.detach() + (features - features.detach())
        quantized = hidden.new_zeros((*mask.shape, q.shape[-1]))
        quantized[mask] = straight_through.to(hidden.dtype)
        indices = torch.full_like(mask, -1, dtype=torch.long)
        indices[mask] = ids
        histogram = torch.bincount(ids, minlength=codes.shape[0])
        perplexity, active = usage_stats(histogram)
        return dict(quantized=quantized, code_ids=indices, code_usage_histogram=histogram,
                    codebook_perplexity=perplexity, active_code_ratio=active,
                    loss_vq_codebook=code_loss, loss_vq_commitment=commit_loss,
                    loss_vq=code_loss + commit_loss)


def usage_stats(counts):
    p = counts.float() / counts.sum().clamp_min(1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum()
    return entropy.exp() * counts.any(), counts.gt(0).float().mean()
