
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from robustbench.utils import load_model

from src.data_loader import CIFARDataLoader
from src.corruptions import apply_corruption
from src.integrated_gradients import integrated_gradients
from src.grad_cam import grad_cam


# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

model = load_model(
    "Standard",
    model_dir="checkpoints",
    dataset="cifar10",
    threat_model="Linf",
).to(device).eval()

target_layer = model.block3.layer[3].conv2
loader = CIFARDataLoader()


# --------------------------------------------------
# 2. Select five development images
# --------------------------------------------------

with open("results/week1_dev_clean_20.csv", newline="") as f:
    pilot = list(csv.DictReader(f))

selected = [
    row for row in pilot
    if row["correct"] == "True"
    and row["source"] == "train"
][:5]

assert len(selected) == 5


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def ig_heatmap(attribution):
    return (
        attribution.abs()
        .sum(dim=1)[0]
        .detach()
        .cpu()
        .numpy()
    )


def normalized_map(array):
    array = np.asarray(array, dtype=np.float64)
    array = np.maximum(array, 0)

    total = array.sum()

    if total <= 1e-12:
        return np.zeros_like(array)

    return array / total


def attribution_loss_gain(clean_map, noisy_map):
    """
    Importance-weighted attribution changes.

    Loss: disappearing attribution weighted by
          importance in the clean map.

    Gain: emerging attribution weighted by
          lower importance in the clean map.
    """

    A = normalized_map(clean_map)
    B = normalized_map(noisy_map)

    # Rescale clean importance to [0, 1]
    importance = A / (A.max() + 1e-12)

    loss_map = importance * np.maximum(A - B, 0)

    gain_map = (
        (1 - importance)
        * np.maximum(B - A, 0)
    )

    return loss_map, gain_map


# --------------------------------------------------
# 4. Prepare figures
# --------------------------------------------------

methods = ["IG", "Grad-CAM"]

figures = {}

for method in methods:
    fig, axes = plt.subplots(
        5, 4,
        figsize=(14, 15),
    )

    figures[method] = (fig, axes)

results = []


# --------------------------------------------------
# 5. Process images
# --------------------------------------------------

for i, row in enumerate(selected):

    image_id = int(row["image_id"])

    clean = loader.get_image(
        image_id,
        source="train",
        device=device,
    )

    noisy = apply_corruption(
        clean,
        "gaussian_noise",
        severity=3,
        seed=image_id,
    )

    with torch.no_grad():
        clean_prob = model(clean).softmax(dim=1)
        noisy_prob = model(noisy).softmax(dim=1)

    target = int(clean_prob.argmax(dim=1).item())
    noisy_prediction = int(noisy_prob.argmax(dim=1).item())

    # Integrated Gradients
    clean_ig = ig_heatmap(
        integrated_gradients(
            model, clean, target, steps=50
        )
    )

    noisy_ig = ig_heatmap(
        integrated_gradients(
            model, noisy, target, steps=50
        )
    )

    # Grad-CAM
    clean_cam = (
        grad_cam(model, clean, target, target_layer)
        [0, 0].detach().cpu().numpy()
    )

    noisy_cam = (
        grad_cam(model, noisy, target, target_layer)
        [0, 0].detach().cpu().numpy()
    )

    clean_image = (
        clean[0].detach().cpu()
        .permute(1, 2, 0).numpy()
    )

    noisy_image = (
        noisy[0].detach().cpu()
        .permute(1, 2, 0).numpy()
    )

    maps = {
        "IG": (clean_ig, noisy_ig),
        "Grad-CAM": (clean_cam, noisy_cam),
    }

    for method, (clean_map, noisy_map) in maps.items():

        loss_map, gain_map = attribution_loss_gain(
            clean_map, noisy_map
        )

        loss_score = float(loss_map.sum())
        gain_score = float(gain_map.sum())

        results.append({
            "image_id": image_id,
            "method": method,
            "severity": 3,
            "clean_prediction": target,
            "noisy_prediction": noisy_prediction,
            "loss_score": loss_score,
            "gain_score": gain_score,
        })

        fig, axes = figures[method]

        axes[i, 0].imshow(clean_image)
        axes[i, 0].set_title(
            f"Image {image_id}\nClean"
        )

        axes[i, 1].imshow(noisy_image)
        axes[i, 1].set_title(
            f"Severity 3\nPred: {noisy_prediction}"
        )

        axes[i, 2].imshow(
            loss_map,
            cmap="Reds",
            vmin=0,
        )
        axes[i, 2].set_title(
            f"Attribution Loss\nL = {loss_score:.4f}"
        )

        axes[i, 3].imshow(
            gain_map,
            cmap="Blues",
            vmin=0,
        )
        axes[i, 3].set_title(
            f"Attribution Gain\nG = {gain_score:.4f}"
        )

        for ax in axes[i]:
            ax.axis("off")

    print(f"Completed image {image_id}")


# --------------------------------------------------
# 6. Save figures
# --------------------------------------------------

for method, (fig, axes) in figures.items():

    fig.suptitle(
        f"{method}: Attribution Loss and Gain "
        "Under Gaussian Noise",
        fontsize=15,
    )

    fig.tight_layout()

    filename = (
        "figures/week1_"
        + method.lower().replace("-", "_")
        + "_loss_gain.png"
    )

    fig.savefig(
        filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", filename)


# --------------------------------------------------
# 7. Save CSV
# --------------------------------------------------

csv_path = "results/week1_attribution_loss_gain.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(results)

print("Saved:", csv_path)
