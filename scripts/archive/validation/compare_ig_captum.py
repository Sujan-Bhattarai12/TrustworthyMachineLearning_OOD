
import torch

from captum.attr import IntegratedGradients
from robustbench.utils import load_model

from src.integrated_gradients import integrated_gradients
from src.data_loader import CIFARDataLoader


# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# Load pretrained WideResNet
model = load_model(
    "Standard",
    model_dir="checkpoints",
    dataset="cifar10",
    threat_model="Linf",
).to(device).eval()


# --------------------------------------------------
# 2. Load development image
# --------------------------------------------------

loader = CIFARDataLoader()

row = loader.get_split("development")[0]

image_id = int(row["image_id"])

image = loader.get_image(
    image_id=image_id,
    source="train",
    device=device,
)

print("Image ID:", image_id)
print("Source:", row["source"])


# --------------------------------------------------
# 3. Choose target class
# --------------------------------------------------

with torch.no_grad():
    target_class = int(
        model(image).argmax(dim=1).item()
    )

steps = 64


# --------------------------------------------------
# 4. Our Integrated Gradients implementation
# --------------------------------------------------

our_ig = integrated_gradients(
    model,
    image,
    target_class,
    steps=steps,
)


# --------------------------------------------------
# 5. Captum Integrated Gradients implementation
# --------------------------------------------------

captum_ig = IntegratedGradients(model).attribute(
    image,
    baselines=torch.zeros_like(image),
    target=target_class,
    n_steps=steps,
    method="riemann_right",
)


# --------------------------------------------------
# 6. Compare attribution values
# --------------------------------------------------

difference = (our_ig - captum_ig).abs()

print("\n--- IG VS CAPTUM ---")

print("Image ID:", image_id)
print("Target class:", target_class)

print("Our IG shape:", tuple(our_ig.shape))
print("Captum IG shape:", tuple(captum_ig.shape))

print("Mean absolute difference:", difference.mean().item())
print("Maximum absolute difference:", difference.max().item())


# --------------------------------------------------
# 7. Numerical tolerance checks
# --------------------------------------------------

print("\n--- NUMERICAL TOLERANCE CHECKS ---")

for tolerance in [1e-4, 2e-4, 5e-4, 1e-3]:

    close = torch.allclose(
        our_ig,
        captum_ig,
        atol=tolerance,
        rtol=tolerance,
    )

    print(f"Tolerance {tolerance}: {close}")


# --------------------------------------------------
# 8. Relative map error
# --------------------------------------------------

relative_error = (
    torch.linalg.vector_norm(
        (our_ig - captum_ig).flatten()
    )
    /
    (
        torch.linalg.vector_norm(
            captum_ig.flatten()
        ) + 1e-8
    )
)

print("\nRelative map error:", relative_error.item())
