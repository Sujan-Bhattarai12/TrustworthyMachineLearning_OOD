import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from robustbench.utils import load_model
from src.integrated_gradients import integrated_gradients

dataset = CIFAR10("./data", train=False, transform=ToTensor())
model = load_model("Standard", "checkpoints", "cifar10", "Linf").eval()

image, label = dataset[0]
attribution = integrated_gradients(
    model, image.unsqueeze(0), label, steps=50
)

heatmap = attribution[0].abs().sum(dim=0).detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(image.permute(1, 2, 0))
axes[0].set_title("Original: Cat")

axes[1].imshow(heatmap, cmap="inferno")
axes[1].set_title("Integrated Gradients")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("figures/archive/ig_cat.png", dpi=200)
plt.show()