
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

csv_path = "normal_skin_dermoscopic_image_ids.csv"
image_folder = Path("dataset/normal_skin")

df = pd.read_csv(csv_path)

# Keep only successfully downloaded images
downloaded = {p.stem for p in image_folder.glob("*.jpg")}
df = df[df["isic_id"].isin(downloaded)].copy()

# Split by patient, not by individual image
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["patient_id"])
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

# Verify patient separation
train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])

assert train_patients.isdisjoint(val_patients)

train_df["class"] = "non_lesional_skin"
val_df["class"] = "non_lesional_skin"

train_df.to_csv("normal_skin_train.csv", index=False)
val_df.to_csv("normal_skin_validation.csv", index=False)

print("NORMAL SKIN SPLIT COMPLETED")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Patient overlap:", len(train_patients & val_patients))
