
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

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

with open("data/cifar-10-batches-py/test_batch", "rb") as f:
    dataset = pickle.load(f, encoding="bytes")

image_id = 0

rgb = dataset[b"data"][image_id].reshape(3, 32, 32)
rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32) / 255.0

image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(image).argmax(dim=1).item()

heatmap = grad_cam(
    model,
    image,
    prediction,
    model.block3.layer[3].conv2,
)

heatmap = heatmap[0, 0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(rgb)
axes[0].set_title("Original Image")

axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
axes[1].set_title("Grad-CAM")

axes[2].imshow(rgb)
axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
axes[2].set_title("Grad-CAM Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("figures/archive/grad_cam_example.png", dpi=200)
plt.show()
