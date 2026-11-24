
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load prediction results
csv_path = "evaluation_results/finetuned_predictions.csv"
df = pd.read_csv(csv_path)

# Select incorrectly classified images
wrong = df[df["correct_prediction"] == False].copy()

print("Total misclassified images:", len(wrong))
print("\nMisclassification summary:")
print(
    wrong.groupby(["actual_class", "predicted_class"])
    .size()
    .to_string()
)

# Create output directory
output_dir = "evaluation_results/misclassified_images"
os.makedirs(output_dir, exist_ok=True)

# Select up to 30 misclassified images
wrong = wrong.head(30)

# Display images in a grid
columns = 3
rows = (len(wrong) + columns - 1) // columns

fig, axes = plt.subplots(
    rows, columns,
    figsize=(15, rows * 4)
)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax, (_, row) in zip(axes, wrong.iterrows()):
    image_path = row["image_path"]

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)

        confidence_columns = [
            "probability_melanoma",
            "probability_nevus",
            "probability_seborrheic_keratosis",
            "probability_non_lesional_skin"
        ]

        confidence = row[confidence_columns].max() * 100

        ax.set_title(
            f"Actual: {row['actual_class']}\n"
            f"Predicted: {row['predicted_class']}\n"
            f"Top probability: {confidence:.1f}%",
            fontsize=9
        )
    else:
        ax.text(0.5, 0.5, "Image not found", ha="center")

    ax.axis("off")

for ax in axes[len(wrong):]:
    ax.axis("off")

plt.tight_layout()

report_path = os.path.join(output_dir, "misclassified_report.png")
plt.savefig(report_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nVisual report saved:", report_path)
