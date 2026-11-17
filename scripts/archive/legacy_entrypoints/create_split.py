
import csv
import pickle
import random
from pathlib import Path

SEED = 42
DEV_POOL_PER_CLASS = 500
SELECTED_DEV_PER_CLASS = 100

DATA_DIR = Path("data/cifar-10-batches-py")
SPLIT_DIR = Path("splits")
SPLIT_DIR.mkdir(exist_ok=True)

# Load labels from the 50,000 CIFAR-10 training images
train_labels = []

for batch_id in range(1, 6):
    with open(DATA_DIR / f"data_batch_{batch_id}", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    train_labels.extend(batch[b"labels"])

assert len(train_labels) == 50000

# Stratified 45,000 training / 5,000 development split
rng = random.Random(SEED)

training_rows = []
development_pool = []

for class_id in range(10):
    ids = [
        i for i, label in enumerate(train_labels)
        if label == class_id
    ]

    rng.shuffle(ids)

    dev_ids = ids[:DEV_POOL_PER_CLASS]
    train_ids = ids[DEV_POOL_PER_CLASS:]

    assert len(dev_ids) == 500
    assert len(train_ids) == 4500

    development_pool.extend(
        {"image_id": i, "class_id": class_id, "source": "train"}
        for i in dev_ids
    )

    training_rows.extend(
        {"image_id": i, "class_id": class_id, "source": "train"}
        for i in train_ids
    )

# Select 100 per class from the 5,000 development pool
selected_dev = []

for class_id in range(10):
    candidates = [
        row for row in development_pool
        if row["class_id"] == class_id
    ]

    chosen = candidates[:SELECTED_DEV_PER_CLASS]

    selected_dev.extend({
        **row,
        "split": "development",
    } for row in chosen)

# Preserve existing 500 final-test image identities
with open(SPLIT_DIR / "split_manifest_v2.csv", newline="") as f:
    previous = list(csv.DictReader(f))

test_rows = [
    {
        "image_id": int(row["image_id"]),
        "class_id": int(row["class_id"]),
        "source": "test",
        "split": "test",
    }
    for row in previous
    if row["split"] == "test"
]

assert len(test_rows) == 500
assert len({r["image_id"] for r in test_rows}) == 500
assert all(
    sum(r["class_id"] == c for r in test_rows) == 50
    for c in range(10)
)

def save_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

# Full S1 split
save_csv(
    SPLIT_DIR / "s1_train_45000.csv",
    training_rows,
    ["image_id", "class_id", "source"],
)

save_csv(
    SPLIT_DIR / "s1_development_5000.csv",
    development_pool,
    ["image_id", "class_id", "source"],
)

# Active P01 experiment manifest
active_rows = selected_dev + test_rows

save_csv(
    SPLIT_DIR / "split_manifest_v2.csv",
    sorted(
        active_rows,
        key=lambda r: (r["source"], r["image_id"]),
    ),
    ["image_id", "class_id", "source", "split"],
)

# Validate no overlap between source training and development
training_ids = {r["image_id"] for r in training_rows}
dev_pool_ids = {r["image_id"] for r in development_pool}
selected_ids = {r["image_id"] for r in selected_dev}

assert training_ids.isdisjoint(dev_pool_ids)
assert selected_ids.issubset(dev_pool_ids)
assert len(selected_ids) == 1000

print("S1 training:", len(training_rows))
print("S1 development pool:", len(development_pool))
print("P01 development:", len(selected_dev))
print("P01 final test:", len(test_rows))
print("Saved: splits/split_manifest_v2.csv")
