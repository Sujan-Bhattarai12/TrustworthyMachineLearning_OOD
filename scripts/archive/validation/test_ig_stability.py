
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from robustbench.utils import load_model

from src.data_loader import CIFARDataLoader
from src.integrated_gradients import integrated_gradients


# ============================================================
# 1. CONFIGURATION
# ============================================================

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

Path("results").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)

steps_list = [16, 32, 64, 128]
reference_steps = 128
num_images = 5


# ============================================================
# 2. LOAD PRETRAINED WIDERESNET
# ============================================================

model = load_model(
    "Standard",
    model_dir="checkpoints",
    dataset="cifar10",
    threat_model="Linf",
).to(device).eval()

loader = CIFARDataLoader()


# ============================================================
# 3. SELECT FIVE CORRECTLY CLASSIFIED IMAGES
# ============================================================

with open("results/validation/week1_dev_clean_20.csv", newline="") as f:
    pilot = list(csv.DictReader(f))

selected = [
    row for row in pilot
    if row["correct"].strip().lower() == "true"
    and row["source"] == "train"
][:num_images]

assert len(selected) == num_images, (
    f"Expected {num_images} correctly classified images, "
    f"but found {len(selected)}."
)


# ============================================================
# 4. HELPER: CONVERT IMAGE FOR DISPLAY
# ============================================================

def image_to_numpy(image):
    """
    Converts [1, 3, H, W] tensor into [H, W, 3].
    Assumes image pixel values are in [0, 1].
    """

    image_np = (
        image[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    return np.clip(image_np, 0, 1)


# ============================================================
# 5. RUN IG NUMERICAL CONVERGENCE TEST
# ============================================================

results = []
visualization_data = []

for row in selected:

    image_id = int(row["image_id"])
    target_class = int(row["predicted_class"])

    image = loader.get_image(
        image_id,
        source="train",
        device=device,
    )

    baseline = torch.zeros_like(image)

    with torch.no_grad():
        output_image = model(image)[0, target_class].item()
        output_baseline = model(baseline)[0, target_class].item()

    logit_difference = output_image - output_baseline

    # Reference attribution: 128 integration steps
    reference = integrated_gradients(
        model,
        image,
        target_class,
        steps=reference_steps,
    ).detach()

    image_heatmaps = {}
    image_metrics = {}

    for steps in steps_list:

        if steps == reference_steps:
            attribution = reference
        else:
            attribution = integrated_gradients(
                model,
                image,
                target_class,
                steps=steps,
            ).detach()

        # ----------------------------------------------------
        # A. COMPLETENESS ERROR
        # ----------------------------------------------------

        completeness_error = abs(
            attribution.sum().item()
            - logit_difference
        )

        # ----------------------------------------------------
        # B. RELATIVE MAP DIFFERENCE
        # ----------------------------------------------------
        # Compare signed IG attribution tensors against
        # the 128-step reference using relative L2 norm.

        numerator = torch.linalg.vector_norm(
            (attribution - reference).flatten()
        )

        denominator = (
            torch.linalg.vector_norm(reference.flatten())
            + 1e-8
        )

        relative_difference = (
            numerator / denominator
        ).item()

        # ----------------------------------------------------
        # C. CREATE ATTRIBUTION HEATMAP
        # ----------------------------------------------------

        heatmap = (
            attribution.abs()
            .sum(dim=1)[0]
            .detach()
            .cpu()
            .numpy()
        )

        image_heatmaps[steps] = heatmap

        image_metrics[steps] = {
            "relative_difference": relative_difference,
            "completeness_error": completeness_error,
        }

        results.append({
            "image_id": image_id,
            "source": "train",
            "target_class": target_class,
            "steps": steps,
            "completeness_error": completeness_error,
            "relative_map_difference": relative_difference,
        })

    visualization_data.append({
        "image_id": image_id,
        "clean_image": image_to_numpy(image),
        "heatmaps": image_heatmaps,
        "metrics": image_metrics,
    })

    print(f"Completed image ID {image_id}")


# ============================================================
# 6. SAVE RESULTS TO CSV
# ============================================================

csv_path = "results/validation/week1_dev_ig_stability_rerun.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {csv_path}")


# ============================================================
# 7. SUMMARIZE NUMERICAL CONVERGENCE
# ============================================================

mean_completeness = []
mean_map_difference = []

print("\n--- IG NUMERICAL CONVERGENCE ---")
print(
    "Steps | Mean completeness error | "
    "Mean relative map difference"
)

for steps in steps_list:

    subset = [
        r for r in results
        if r["steps"] == steps
    ]

    comp = np.mean([
        r["completeness_error"]
        for r in subset
    ])

    diff = np.mean([
        r["relative_map_difference"]
        for r in subset
    ])

    mean_completeness.append(comp)
    mean_map_difference.append(diff)

    print(
        f"{steps:5d} | "
        f"{comp:23.6f} | "
        f"{diff:.6f}"
    )


# ============================================================
# 8. PLOT NUMERICAL CONVERGENCE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    steps_list,
    mean_map_difference,
    marker="o",
    linewidth=2,
    markersize=7,
)

ax.set_xlabel("IG integration steps")
ax.set_ylabel("Mean relative difference from 128-step IG")
ax.set_title("Integrated Gradients Numerical Convergence")
ax.set_xticks(steps_list)
ax.grid(alpha=0.3)

plt.tight_layout()

convergence_path = "figures/validation/week1_dev_ig_stability_rerun.png"

plt.savefig(
    convergence_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved {convergence_path}")


# ============================================================
# 9. VISUALIZE CLEAN IMAGES + IG HEATMAPS + METRICS
# ============================================================

n_rows = len(visualization_data)
n_cols = len(steps_list) + 1

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(19, 16),
    squeeze=False,
)

for i, item in enumerate(visualization_data):

    clean_image = item["clean_image"]
    heatmaps = item["heatmaps"]
    metrics = item["metrics"]

    # --------------------------------------------------------
    # FIRST COLUMN: ORIGINAL CLEAN IMAGE
    # --------------------------------------------------------

    ax = axes[i, 0]

    ax.imshow(
        clean_image,
        interpolation="nearest",
    )

    ax.set_title(
        "Clean image",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        f"Image {item['image_id']}",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # --------------------------------------------------------
    # HEATMAP COLUMNS
    # --------------------------------------------------------

    # Consistent color scale within each image
    vmax = max(
        float(h.max())
        for h in heatmaps.values()
    )

    if vmax <= 0:
        vmax = 1.0

    for j, steps in enumerate(steps_list, start=1):

        ax = axes[i, j]

        ax.imshow(
            heatmaps[steps],
            cmap="hot",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        difference = metrics[steps]["relative_difference"]
        completeness = metrics[steps]["completeness_error"]

        # Relative difference displayed above heatmap
        ax.set_title(
            f"{steps} steps\n"
            f"Map diff: {difference:.4f}",
            fontsize=10,
            fontweight="bold",
        )

        # Completeness error displayed below heatmap
        ax.set_xlabel(
            f"Completeness error: {completeness:.4f}",
            fontsize=9,
        )

        ax.set_xticks([])
        ax.set_yticks([])


fig.suptitle(
    "Integrated Gradients Numerical Convergence\n"
    "Clean Images, Attribution Maps, and Numerical Errors",
    fontsize=17,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96],
    h_pad=2.5,
    w_pad=1.5,
)

heatmap_path = (
    "figures/validation/week1_dev_ig_convergence_heatmaps_rerun.png"
)

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved {heatmap_path}")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IG NUMERICAL CONVERGENCE TEST COMPLETED")
print("=" * 60)

print(f"Images evaluated: {num_images}")
print(f"Integration steps: {steps_list}")
print(f"Reference: {reference_steps} steps")

print("\nGenerated files:")
print(f"1. {csv_path}")
print(f"2. {convergence_path}")
print(f"3. {heatmap_path}")

print("\nDone.")
