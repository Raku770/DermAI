
import pandas as pd

# Load ground-truth CSV
csv_path = "dataset/ISIC-2017_Training_Part3_GroundTruth.csv"

df = pd.read_csv(csv_path)

# Create class labels
def assign_class(row):
    if row["melanoma"] == 1:
        return "melanoma"
    elif row["seborrheic_keratosis"] == 1:
        return "seborrheic_keratosis"
    else:
        return "nevus"

df["class"] = df.apply(assign_class, axis=1)

# Display class distribution
print("Class distribution:")
print(df["class"].value_counts())

# Check unexpected combinations
print("\nLabel combinations:")
print(
    df[["melanoma", "seborrheic_keratosis"]]
    .value_counts()
)

# Display first five rows
print("\nSample labels:")
print(df[["image_id", "class"]].head())