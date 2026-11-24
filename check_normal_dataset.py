
import pandas as pd
from pathlib import Path

csv_path = "normal_skin_dermoscopic_image_ids.csv"
image_folder = Path("dataset/normal_skin")

df = pd.read_csv(csv_path)

downloaded_ids = {
    p.stem for p in image_folder.glob("*.jpg")
}

df["downloaded"] = df["isic_id"].isin(downloaded_ids)

print("Total CSV records:", len(df))
print("Downloaded images:", df["downloaded"].sum())
print("Unique patients:", df.loc[df["downloaded"], "patient_id"].nunique())
print("\nImages per patient:")
print(
    df[df["downloaded"]]
    .groupby("patient_id")["isic_id"]
    .nunique()
    .describe()
)

print("\nAnatomical site distribution:")
print(df[df["downloaded"]]["anatom_site_general"].value_counts(dropna=False))
