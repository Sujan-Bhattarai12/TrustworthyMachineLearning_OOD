
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from captum.attr import LayerGradCam
from robustbench.utils import load_model
from src.grad_cam import grad_cam

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = load_model(
    "Standard",
    model_dir="checkpoints",
    dataset="cifar10",
    threat_model="Linf",
).to(device).eval()

from src.data_loader import CIFARDataLoader

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

with torch.no_grad():
    target_class = model(image).argmax(dim=1).item()

target_layer = model.block3.layer[3].conv2

# Our implementation
our_cam = grad_cam(
    model, image, target_class, target_layer
)

# Captum implementation
captum_cam = LayerGradCam(
    model,
    target_layer,
).attribute(
    image,
    target=target_class,
    relu_attributions=True,
)

# Apply the same upsampling as our implementation
captum_cam = F.interpolate(
    captum_cam,
    size=image.shape[-2:],
    mode="bilinear",
    align_corners=False,
)

# Apply the same normalization
cam_min = captum_cam.amin(dim=(2, 3), keepdim=True)
cam_max = captum_cam.amax(dim=(2, 3), keepdim=True)

captum_cam = (
    (captum_cam - cam_min)
    / (cam_max - cam_min + 1e-8)
)

# Numerical comparison
difference = (our_cam - captum_cam).abs()

print("Target class:", target_class)
print("Our CAM shape:", tuple(our_cam.shape))
print("Captum CAM shape:", tuple(captum_cam.shape))
print("Mean absolute difference:", difference.mean().item())
print("Maximum absolute difference:", difference.max().item())
print(
    "Numerically close:",
    torch.allclose(our_cam, captum_cam, atol=1e-4, rtol=1e-4),
)
