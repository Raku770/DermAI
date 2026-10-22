
import os
import pandas as pd

dataset_path = "dataset"

csv_files = [
    file for file in os.listdir(dataset_path)
    if file.lower().endswith(".csv")
]

image_files = [
    file for file in os.listdir(dataset_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("CSV files found:", csv_files)
print("Number of images:", len(image_files))

for csv_file in csv_files:
    csv_path = os.path.join(dataset_path, csv_file)
    df = pd.read_csv(csv_path)

    print("\nCSV:", csv_file)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())