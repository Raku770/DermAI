
import pandas as pd

csv_path = "evaluation_results/finetuned_predictions.csv"
df = pd.read_csv(csv_path)

probability_columns = [
    "probability_melanoma",
    "probability_nevus",
    "probability_seborrheic_keratosis",
    "probability_non_lesional_skin"
]

# Calculate prediction confidence
df["prediction_confidence"] = df[probability_columns].max(axis=1)

# Get probability assigned to the actual class
class_to_column = {
    "melanoma": "probability_melanoma",
    "nevus": "probability_nevus",
    "seborrheic_keratosis": "probability_seborrheic_keratosis",
    "non_lesional_skin": "probability_non_lesional_skin"
}

df["actual_class_probability"] = df.apply(
    lambda row: row[class_to_column[row["actual_class"]]],
    axis=1
)

wrong = df[df["correct_prediction"] == False].copy()

print("\nMISCLASSIFICATION CONFIDENCE")
print("--------------------------------")

print("Total incorrect:", len(wrong))

for threshold in [0.70, 0.80, 0.90]:
    count = (wrong["prediction_confidence"] >= threshold).sum()
    print(f"Incorrect predictions with confidence >= {threshold:.0%}: {count}")

print("\nAverage confidence of incorrect predictions:")
print(f"{wrong['prediction_confidence'].mean():.2%}")

print("\nConfusion-wise confidence:")
print(
    wrong.groupby(["actual_class", "predicted_class"])
    ["prediction_confidence"]
    .agg(["count", "mean", "max"])
    .round(3)
    .to_string()
)

# Save highest-confidence mistakes
wrong = wrong.sort_values(
    "prediction_confidence",
    ascending=False
)

output_path = "evaluation_results/high_confidence_errors.csv"

wrong.to_csv(output_path, index=False)

print("\nSaved:", output_path)
