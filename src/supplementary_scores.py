"""Deterministic Shannon predictive entropy and Liu et al. (2020) energy."""
import math
import torch


def supplementary_scores(logits, temperature=1.0):
    """Higher scores indicate warning; entropy uses unscaled probabilities."""
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite batch-by-class logits with at least two classes.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Energy temperature must be finite and positive.')
    logp = torch.log_softmax(logits.double(), dim=1)
    return {'Entropy': -(logp.exp() * logp).sum(dim=1),
            'Energy': -temperature * torch.logsumexp(logits.double() / temperature, dim=1)}
