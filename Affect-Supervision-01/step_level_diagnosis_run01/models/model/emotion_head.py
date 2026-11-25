import torch
from torch import nn
from torch.nn import functional as F


class EmotionHead(nn.Module):
    def __init__(self, dim, mlp_dim, classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, classes))

    def forward(self, hidden, mask):
        mask = mask.bool().unsqueeze(-1)
        pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


def emotion_loss(logits, labels):
    # GLIM.encode_labels uses -1 for missing ZuCo labels.
    labels = labels.masked_fill(labels < 0, -100)
    return F.cross_entropy(logits.float(), labels.long(), ignore_index=-100,
                           reduction="sum") / labels.ge(0).sum().clamp_min(1)

class EmotionAttribute(nn.Module):
    def __init__(self, dim, memory_dim, temperature, gate_init, detach_probs):
        super().__init__()
        if temperature <= 0 or not 0 < gate_init < 1:
            raise ValueError("Require temperature > 0 and 0 < gate_init < 1")
        self.temperature = temperature
        self.detach_probs = detach_probs
        self.embedding = nn.Embedding(3, dim)
        self.projection = nn.Linear(dim, memory_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)).logit())

    def forward(self, logits):
        probs = (logits.float() / self.temperature).softmax(-1)
        if self.detach_probs:
            probs = probs.detach()
        return self.gate.sigmoid() * self.projection(probs @ self.embedding.weight)
