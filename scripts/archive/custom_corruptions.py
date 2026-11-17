
import torch
import torch.nn.functional as F


def apply_corruption(image, corruption, severity, seed=42):
    """
    Apply image corruption at a specified severity level.

    Parameters
    ----------
    image : torch.Tensor
        Shape (1, 3, 32, 32), values in [0, 1].
    corruption : str
        "gaussian_noise", "brightness", or "blur".
    severity : int
        Integer from 0 to 5.
        0 means the original, uncorrupted image.
    seed : int
        Random seed for reproducible Gaussian noise.

    Returns
    -------
    torch.Tensor
        Corrupted image with the same shape as the input.
    """

    if severity not in (0, 1, 2, 3, 4, 5):
        raise ValueError("Severity must be between 0 and 5")

    # Severity 0: original image
    if severity == 0:
        return image.clone()

    if corruption == "gaussian_noise":

        std = {
            1: 0.04,
            2: 0.08,
            3: 0.12,
            4: 0.16,
            5: 0.20,
        }[severity]

        # Local generator for reproducible noise
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        noise = torch.randn(
            image.shape,
            generator=generator,
            dtype=image.dtype,
        ).to(image.device)

        return (image + std * noise).clamp(0, 1)

    elif corruption == "brightness":

        amount = {
            1: 0.10,
            2: 0.175,
            3: 0.25,
            4: 0.325,
            5: 0.40,
        }[severity]

        return (image + amount).clamp(0, 1)

    elif corruption == "blur":

        sigma = {
            1: 0.5,
            2: 0.75,
            3: 1.0,
            4: 1.25,
            5: 1.5,
        }[severity]

        radius = 3

        coords = torch.arange(
            -radius,
            radius + 1,
            device=image.device,
            dtype=image.dtype,
        )

        kernel_1d = torch.exp(
            -(coords ** 2) / (2 * sigma ** 2)
        )

        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)

        kernel = kernel_2d.expand(3, 1, 7, 7)

        padded = F.pad(
            image,
            (3, 3, 3, 3),
            mode="reflect",
        )

        return F.conv2d(
            padded,
            kernel,
            groups=3,
        ).clamp(0, 1)

    else:
        raise ValueError(
            f"Unknown corruption: {corruption}"
        )


def apply_corruption_strength(image, corruption, strength, seed=42):
    """Continuous development corruption: noise std or additive brightness.

    Kept separate so historical integer-severity experiments remain unchanged.
    """
    import math
    maximum = {'gaussian_noise': 0.20, 'brightness': 0.40}
    if corruption not in maximum:
        raise ValueError('Continuous study supports development noise/brightness only.')
    if not math.isfinite(strength) or not 0 <= strength <= maximum[corruption]:
        raise ValueError('Strength is outside the original corruption range.')
    if strength == 0:
        return image.clone()
    if corruption == 'brightness':
        return (image + strength).clamp(0, 1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype).to(image.device)
    return (image + strength * noise).clamp(0, 1)
