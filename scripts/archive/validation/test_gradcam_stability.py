
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from robustbench.utils import load_model

from src.data_loader import CIFARDataLoader
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

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

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

with open("results/validation/week1_dev_clean_20.csv", newline="") as f:
    pilot = list(csv.DictReader(f))

selected = [
    row for row in pilot
    if row["correct"] == "True"
    and row["source"] == "train"
][:5]

assert len(selected) == 5

noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.07]

results = []


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def normalize_map(array):
    return array / (array.sum() + 1e-8)


def generate_noise(image, sigma, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    noise = torch.randn(
        image.shape,
        generator=generator,
        dtype=image.dtype,
    ).to(image.device)

    return (image + sigma * noise).clamp(0, 1)


# --------------------------------------------------
# 4. Prepare Grad-CAM visualization
# --------------------------------------------------

fig_maps, axes_maps = plt.subplots(
    len(selected),
    len(noise_levels) + 1,
    figsize=(15, 13),
)

# --------------------------------------------------
# 5. Run Grad-CAM stability experiment
# --------------------------------------------------

for i, row in enumerate(selected):

    image_id = int(row["image_id"])

    image = loader.get_image(
        image_id,
        source="train",
        device=device,
    )

    with torch.no_grad():
        clean_logits = model(image)
        target_class = int(
            clean_logits.argmax(dim=1).item()
        )

    clean_cam = grad_cam(
        model,
        image,
        target_class,
        target_layer,
    )[0, 0].detach().cpu().numpy()

    clean_map = normalize_map(clean_cam)

    # Original image
    axes_maps[i, 0].imshow(
        image[0].detach().cpu()
        .permute(1, 2, 0).numpy()
    )
    axes_maps[i, 0].set_title(
        f"Image {image_id}\nClean"
    )
    axes_maps[i, 0].axis("off")

    for j, sigma in enumerate(noise_levels):

        noisy_image = generate_noise(
            image,
            sigma,
            seed=image_id,
        )

        with torch.no_grad():
            noisy_logits = model(noisy_image)
            noisy_prediction = int(
                noisy_logits.argmax(dim=1).item()
            )

        noisy_cam = grad_cam(
            model,
            noisy_image,
            target_class,
            target_layer,
        )[0, 0].detach().cpu().numpy()

        noisy_map = normalize_map(noisy_cam)

        map_difference = float(
            np.abs(clean_map - noisy_map).sum() / 2
        )

        prediction_unchanged = (
            noisy_prediction == target_class
        )

        results.append({
            "image_id": image_id,
            "noise_sigma": sigma,
            "clean_prediction": target_class,
            "noisy_prediction": noisy_prediction,
            "prediction_unchanged": prediction_unchanged,
            "cam_map_difference": map_difference,
        })

        # Display Grad-CAM heatmap
        axes_maps[i, j + 1].imshow(
            noisy_cam,
            cmap="hot",
            vmin=0,
            vmax=1,
        )

        axes_maps[i, j + 1].set_title(
            f"Noise {sigma}\n"
            f"Diff {map_difference:.3f}"
        )

        axes_maps[i, j + 1].axis("off")

    print(f"Completed image {image_id}")


# --------------------------------------------------
# 6. Save Grad-CAM heatmaps PNG
# --------------------------------------------------

fig_maps.suptitle(
    "Grad-CAM Stability Across Small Perturbations",
    fontsize=15,
)

fig_maps.tight_layout()

fig_maps.savefig(
    "figures/validation/week1_dev_gradcam_heatmaps_rerun.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig_maps)


# --------------------------------------------------
# 7. Save CSV
# --------------------------------------------------

csv_path = "results/validation/week1_dev_gradcam_stability_rerun.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 8. Calculate summary statistics
# --------------------------------------------------

mean_differences = []

print("\n--- GRAD-CAM STABILITY ---")
print("Noise | Mean map difference | Unchanged predictions")

for sigma in noise_levels:

    subset = [
        row for row in results
        if row["noise_sigma"] == sigma
    ]

    mean_diff = float(np.mean([
        row["cam_map_difference"]
        for row in subset
    ]))

    unchanged = sum(
        row["prediction_unchanged"]
        for row in subset
    )

    mean_differences.append(mean_diff)

    print(
        f"{sigma:5.3f} | "
        f"{mean_diff:.4f} | "
        f"{unchanged}/{len(subset)}"
    )


# --------------------------------------------------
# 9. Save stability graph PNG
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    noise_levels,
    mean_differences,
    marker="o",
    linewidth=2,
)

ax.set_xlabel("Gaussian Noise Standard Deviation")
ax.set_ylabel("Mean Grad-CAM Map Difference")
ax.set_title("Grad-CAM Stability Under Small Perturbations")

ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    "figures/validation/week1_dev_gradcam_stability_rerun.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# --------------------------------------------------
# 10. Final output
# --------------------------------------------------

print("\nSaved files:")
print(csv_path)
print("figures/validation/week1_dev_gradcam_stability_rerun.png")
print("figures/validation/week1_dev_gradcam_heatmaps_rerun.png")
