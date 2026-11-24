
import requests
import json

image_id = "ISIC_0000095"

url = f"https://api.isic-archive.com/api/v2/images/{image_id}/"

response = requests.get(url, timeout=60)
response.raise_for_status()

data = response.json()

print("Image ID:", image_id)
print("Available metadata fields:")
print(json.dumps(data, indent=2)[:6000])
