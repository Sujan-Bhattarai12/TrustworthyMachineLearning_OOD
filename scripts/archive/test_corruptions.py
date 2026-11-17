
import matplotlib.pyplot as plt

from src.data_loader import CIFARDataLoader
from src.corruptions import apply_corruption


loader = CIFARDataLoader()

row = loader.get_split("development")[0]

image = loader.get_image(
    image_id=int(row["image_id"]),
    source=row["source"],
)

corruptions = ["gaussian_noise", "brightness", "blur"]
severities = [1, 3, 5]

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, corruption in enumerate(corruptions):
    axes[i, 0].imshow(
        image[0].permute(1, 2, 0).numpy()
    )
    axes[i, 0].set_title("Clean")

    for j, severity in enumerate(severities, start=1):
        corrupted = apply_corruption(
            image,
            corruption,
            severity,
            seed=int(row["image_id"]),
        )

        axes[i, j].imshow(
            corrupted[0].permute(1, 2, 0).numpy()
        )
        axes[i, j].set_title(
            f"{corruption}\nSeverity {severity}"
        )

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(
    "figures/examples/development_corruptions.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

print("Development image ID:", row["image_id"])
print("Generated 9 corrupted versions successfully.")
print("Saved: figures/examples/development_corruptions.png")
