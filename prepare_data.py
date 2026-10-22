import os
import pandas as pd

# Paths
dataset_path = "dataset"
csv_path = os.path.join(
    dataset_path,
    "ISIC-2017_Training_Part3_GroundTruth.csv"
)

# Read ground-truth CSV
df = pd.read_csv(csv_path)

# Convert binary labels into 3 classes
def assign_class(row):
    if row["melanoma"] == 1:
        return "melanoma"
    elif row["seborrheic_keratosis"] == 1:
        return "seborrheic_keratosis"
    else:
        return "nevus"

df["class"] = df.apply(assign_class, axis=1)

# Create image path
df["image_path"] = df["image_id"].apply(
    lambda x: os.path.join(dataset_path, x + ".jpg")
)

# Check whether each image exists
df["image_exists"] = df["image_path"].apply(os.path.exists)

# Results
print("Total records:", len(df))
print("Images found:", df["image_exists"].sum())
print("Images missing:", (~df["image_exists"]).sum())

print("\nClass distribution:")
print(df["class"].value_counts())

print("\nFirst 5 records:")
print(df[["image_id", "image_path", "class"]].head())

# Save prepared dataset
df.to_csv("prepared_dataset.csv", index=False)

print("\nPrepared dataset saved as: prepared_dataset.csv")