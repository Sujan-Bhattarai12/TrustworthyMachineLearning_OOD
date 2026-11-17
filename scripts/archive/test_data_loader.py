
from src.data_loader import CIFARDataLoader

loader = CIFARDataLoader()

development = loader.get_split("development")
test = loader.get_split("test")

print("Development images:", len(development))
print("Test images:", len(test))

# Retrieve the first development image
dev_row = development[0]

dev_image = loader.get_image(
    image_id=int(dev_row["image_id"]),
    source=dev_row["source"],
)

# Retrieve the first final-test image
test_row = test[0]

test_image = loader.get_image(
    image_id=int(test_row["image_id"]),
    source=test_row["source"],
)

print("\nDevelopment example:")
print("Image ID:", dev_row["image_id"])
print("Source:", dev_row["source"])
print("Shape:", tuple(dev_image.shape))
print("Pixel range:", dev_image.min().item(), dev_image.max().item())

print("\nTest example:")
print("Image ID:", test_row["image_id"])
print("Source:", test_row["source"])
print("Shape:", tuple(test_image.shape))
print("Pixel range:", test_image.min().item(), test_image.max().item())

assert len(development) == 1000
assert len(test) == 500
assert all(row["source"] == "train" for row in development)
assert all(row["source"] == "test" for row in test)

print("\nData loader verification PASSED")
