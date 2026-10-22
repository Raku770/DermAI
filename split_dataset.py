import pandas as pd
from sklearn.model_selection import train_test_split

# Load the prepared dataset
df = pd.read_csv("prepared_dataset.csv")

# Split into 80% training and 20% validation
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)

# Save the two datasets
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)

# Display results
print("Total images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["class"].value_counts())

print("\nValidation class distribution:")
print(val_df["class"].value_counts())