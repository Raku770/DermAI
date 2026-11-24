
import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

csv_path = "normal_skin_dermoscopic_image_ids.csv"
output_dir = Path("dataset/normal_skin")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
image_ids = df["isic_id"].dropna().astype(str).unique()

session = requests.Session()

success = 0
failed = []

print("Total image IDs:", len(image_ids))

for index, image_id in enumerate(image_ids, start=1):
    output_path = output_dir / f"{image_id}.jpg"

    if output_path.exists():
        success += 1
        continue

    try:
        metadata_url = f"https://api.isic-archive.com/api/v2/images/{image_id}/"
        response = session.get(metadata_url, timeout=60)
        response.raise_for_status()

        image_url = response.json()["files"]["full"]["url"]

        image_response = session.get(image_url, timeout=60)
        image_response.raise_for_status()

        image = Image.open(BytesIO(image_response.content)).convert("RGB")
        image.save(output_path, quality=95)

        success += 1

    except Exception as e:
        failed.append((image_id, str(e)))

    if index % 25 == 0:
        print(f"Progress: {index}/{len(image_ids)} | Saved: {success} | Failed: {len(failed)}")

    time.sleep(0.2)

pd.DataFrame(failed, columns=["isic_id", "error"]).to_csv(
    "normal_skin_failed_downloads.csv", index=False
)

print("\nDOWNLOAD COMPLETED")
print("Saved images:", success)
print("Failed images:", len(failed))
print("Failed IDs saved to normal_skin_failed_downloads.csv")
