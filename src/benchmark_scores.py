"""Warning baselines. Inputs are raw [0,1] images; normalization belongs in model.

ODIN: https://arxiv.org/abs/1706.02690 (temperature + confidence-increasing step).
All returned warning scores increase with uncertainty.
"""
import torch
from torch.nn import functional as F


def odin(model, image, temperature=1000.0, epsilon=0.0014):
    """ODIN in raw pixel coordinates; clip perturbed image to [0,1].

    This explicit bounded-pixel adaptation differentiates through model preprocessing.
    T and epsilon are pilot defaults, not selected/tuned on held-out test data.
    """
    if temperature <= 0 or epsilon < 0:
        raise ValueError('ODIN requires positive temperature and nonnegative epsilon.')
    states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.enable_grad():
            x = image.detach().clone().requires_grad_(True)
            logits = model(x)
            target = logits.detach().argmax(dim=1)
            loss = F.cross_entropy(logits / temperature, target)
            gradient = torch.autograd.grad(loss, x)[0]
            perturbed = (x - epsilon * gradient.sign()).clamp(0, 1).detach()
        with torch.no_grad():
            return 1 - (model(perturbed) / temperature).softmax(dim=1).max(dim=1).values
    finally:
        for module, training in states:
            module.training = training


def benchmark_scores(model, image, temperature=1000.0, epsilon=0.0014):
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probabilities = logits.softmax(dim=1)
    scores = {}
    scores['odin'] = odin(model, image, temperature, epsilon)
    scores['msp_warning'] = 1 - probabilities.max(dim=1).values
    return logits, scores
