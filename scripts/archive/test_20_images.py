
import csv
import time
from pathlib import Path

import torch
from robustbench.utils import load_model

from src.data_loader import CIFARDataLoader
from src.integrated_gradients import integrated_gradients
from src.grad_cam import grad_cam


# 1. Set up device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Device:", device)


# 2. Load frozen WideResNet
model = load_model(
    "Standard",
    model_dir="checkpoints",
    dataset="cifar10",
    threat_model="Linf",
)

model = model.to(device).eval()

target_layer = model.block3.layer[3].conv2


# 3. Load correct development images
loader = CIFARDataLoader()
development = loader.get_split("development")

# Select two images per class, 20 total
selected = []

for class_id in range(10):
    candidates = [
        row for row in development
        if int(row["class_id"]) == class_id
    ]
    selected.extend(candidates[:2])

assert len(selected) == 20
assert all(row["source"] == "train" for row in selected)


# 4. Run predictions and explanations
results = []

for index, row in enumerate(selected, start=1):
    image_id = int(row["image_id"])
    true_class = int(row["class_id"])

    image = loader.get_image(
        image_id=image_id,
        source="train",
        device=device,
    )

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities.max().item())

    correct = predicted_class == true_class

    # Integrated Gradients
    start = time.perf_counter()

    ig = integrated_gradients(
        model,
        image,
        target_class=predicted_class,
        steps=50,
    )

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    ig_time = time.perf_counter() - start

    # Grad-CAM
    start = time.perf_counter()

    cam = grad_cam(
        model,
        image,
        target_class=predicted_class,
        target_layer=target_layer,
    )

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    cam_time = time.perf_counter() - start

    ig_valid = bool(torch.isfinite(ig).all().item())
    cam_valid = bool(torch.isfinite(cam).all().item())

    results.append({
        "image_id": image_id,
        "source": "train",
        "true_class": true_class,
        "predicted_class": predicted_class,
        "correct": correct,
        "confidence": confidence,
        "ig_valid": ig_valid,
        "cam_valid": cam_valid,
        "ig_time_seconds": ig_time,
        "cam_time_seconds": cam_time,
    })

    print(
        f"{index:02d}/20 | "
        f"ID={image_id} | "
        f"True={true_class} | "
        f"Pred={predicted_class} | "
        f"Correct={correct} | "
        f"Confidence={confidence:.4f}"
    )


# 5. Save results
Path("results").mkdir(exist_ok=True)

output_path = "results/week1_dev_clean_20.csv"

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# 6. Summary
accuracy = sum(row["correct"] for row in results) / len(results)

mean_ig_time = sum(
    row["ig_time_seconds"] for row in results
) / len(results)

mean_cam_time = sum(
    row["cam_time_seconds"] for row in results
) / len(results)

print("\n--- WEEK 1 DEVELOPMENT PILOT ---")
print(f"Accuracy: {accuracy:.1%}")
print(f"Valid IG maps: {sum(r['ig_valid'] for r in results)}/20")
print(f"Valid Grad-CAM maps: {sum(r['cam_valid'] for r in results)}/20")
print(f"Mean IG time: {mean_ig_time:.3f} seconds")
print(f"Mean Grad-CAM time: {mean_cam_time:.3f} seconds")
print(f"Saved: {output_path}")
