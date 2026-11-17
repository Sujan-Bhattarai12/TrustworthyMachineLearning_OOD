import torch

def integrated_gradients(model, image, target_class, steps=64):
    if steps not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError('IG steps must be a power of two from 1 to 128.')
    baseline = torch.zeros_like(image)
    total_gradients = torch.zeros_like(image)

    for i in range(1, steps + 1):
        alpha = i / steps
        x = (baseline + alpha * (image - baseline)).detach()
        x.requires_grad_(True)

        output = model(x)
        score = output[:, target_class].sum()

        gradient = torch.autograd.grad(score, x)[0]
        total_gradients += gradient

    return (image - baseline) * total_gradients / steps