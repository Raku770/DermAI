
import pandas as pd
from pathlib import Path

# Load original datasets
old_train = pd.read_csv("train.csv")
old_val = pd.read_csv("validation.csv")

# Load new non-lesional datasets
normal_train = pd.read_csv("normal_skin_train.csv")
normal_val = pd.read_csv("normal_skin_validation.csv")

# Check for image ID overlap
old_ids = set(old_train["image_id"]) | set(old_val["image_id"])
normal_ids = set(normal_train["isic_id"]) | set(normal_val["isic_id"])

overlap = old_ids.intersection(normal_ids)

if overlap:
    raise ValueError(f"Duplicate image IDs found: {len(overlap)}")

# Prepare new image paths
normal_train["image_id"] = normal_train["isic_id"]
normal_val["image_id"] = normal_val["isic_id"]

normal_train["image_path"] = normal_train["isic_id"].apply(
    lambda x: str(Path("dataset/normal_skin") / f"{x}.jpg")
)
normal_val["image_path"] = normal_val["isic_id"].apply(
    lambda x: str(Path("dataset/normal_skin") / f"{x}.jpg")
)

normal_train["image_exists"] = normal_train["image_path"].apply(
    lambda x: Path(x).exists()
)
normal_val["image_exists"] = normal_val["image_path"].apply(
    lambda x: Path(x).exists()
)

# Keep only the required common columns
columns = ["image_id", "class", "image_path", "image_exists"]

train_four = pd.concat(
    [old_train[columns], normal_train[columns]],
    ignore_index=True
)

val_four = pd.concat(
    [old_val[columns], normal_val[columns]],
    ignore_index=True
)

# Verify all image paths
assert train_four["image_exists"].all()
assert val_four["image_exists"].all()

# Save as separate files
train_four.to_csv("train_four_class.csv", index=False)
val_four.to_csv("validation_four_class.csv", index=False)

print("FOUR-CLASS DATASET PREPARED")
print("\nTraining distribution:")
print(train_four["class"].value_counts())

print("\nValidation distribution:")
print(val_four["class"].value_counts())

print("\nTotal training:", len(train_four))
print("Total validation:", len(val_four))
print("Missing training images:", (~train_four["image_exists"]).sum())
print("Missing validation images:", (~val_four["image_exists"]).sum())