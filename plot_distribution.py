import pandas as pd
import matplotlib.pyplot as plt

# Load ground-truth CSV
csv_path = "dataset/ISIC-2017_Training_Part3_GroundTruth.csv"
df = pd.read_csv(csv_path)

# Convert binary labels into 3 classes
def assign_class(row):
    if row["melanoma"] == 1:
        return "Melanoma"
    elif row["seborrheic_keratosis"] == 1:
        return "Seborrheic Keratosis"
    else:
        return "Nevus"

df["class"] = df.apply(assign_class, axis=1)

# Count images in each class
class_counts = df["class"].value_counts()

print("Class distribution:")
print(class_counts)

# Create bar chart
plt.figure(figsize=(8, 5))

class_counts.plot(kind="bar")

plt.title("ISIC 2017 Class Distribution")
plt.xlabel("Skin Lesion Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)

plt.tight_layout()

# Save the graph
plt.savefig("class_distribution.png", dpi=300)

# Show graph
plt.show()