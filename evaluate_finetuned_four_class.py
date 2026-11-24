
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

# ---------------------------------
# 1. Configuration
# ---------------------------------

CLASSES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis",
    "non_lesional_skin"
]

IMG_SIZE = 224
BATCH_SIZE = 8

MODEL_PATH = Path(
    "models/efficientnet_b0_four_class_finetuned_best.keras"
)

CSV_PATH = Path("validation_four_class.csv")

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------
# 2. Verify files
# ---------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

val_df = pd.read_csv(CSV_PATH)

assert val_df["image_path"].notna().all()
assert val_df["class"].notna().all()

assert val_df["class"].isin(CLASSES).all()

assert val_df["image_path"].apply(
    lambda p: Path(p).is_file()
).all()

print("Validation images:", len(val_df))
print("\nValidation distribution:")
print(val_df["class"].value_counts())

# ---------------------------------
# 3. Load fine-tuned model
# ---------------------------------

print("\nLoading fine-tuned model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Fine-tuned model loaded successfully!")

# ---------------------------------
# 4. Prepare validation dataset
# ---------------------------------

class_map = {
    name: i for i, name in enumerate(CLASSES)
}

labels = val_df["class"].map(class_map).to_numpy(
    dtype=np.int32
)

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].astype(str).to_numpy(),
        labels
    )
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# ---------------------------------
# 5. Generate predictions
# ---------------------------------

print("\nEvaluating fine-tuned four-class model...")

predictions = model.predict(
    dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

# ---------------------------------
# 6. Overall accuracy
# ---------------------------------

accuracy = accuracy_score(
    labels,
    predicted_labels
)

print("\nOVERALL VALIDATION ACCURACY:")
print(f"{accuracy * 100:.2f}%")

# ---------------------------------
# 7. Classification report
# ---------------------------------

report = classification_report(
    labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    target_names=CLASSES,
    zero_division=0,
    output_dict=True
)

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        labels,
        predicted_labels,
        labels=[0, 1, 2, 3],
        target_names=CLASSES,
        zero_division=0
    )
)

# Save classification report
report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_DIR / "finetuned_classification_report.csv"
)

# ---------------------------------
# 8. Confusion matrix
# ---------------------------------

cm = confusion_matrix(
    labels,
    predicted_labels,
    labels=[0, 1, 2, 3]
)

print("\nCONFUSION MATRIX:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=30,
    values_format="d"
)

plt.title("DermAI Fine-Tuned Four-Class Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "finetuned_confusion_matrix.png",
    dpi=300
)

plt.close()

# ---------------------------------
# 9. Save prediction results
# ---------------------------------

results_df = val_df.copy()

results_df["actual_class"] = [
    CLASSES[i] for i in labels
]

results_df["predicted_class"] = [
    CLASSES[i] for i in predicted_labels
]

results_df["correct_prediction"] = (
    labels == predicted_labels
)

for i, class_name in enumerate(CLASSES):

    results_df[f"probability_{class_name}"] = (
        predictions[:, i]
    )

results_df.to_csv(
    OUTPUT_DIR / "finetuned_predictions.csv",
    index=False
)

# ---------------------------------
# 10. Final summary
# ---------------------------------

print("\nEVALUATION COMPLETED!")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nGenerated files:")
print(OUTPUT_DIR / "finetuned_classification_report.csv")
print(OUTPUT_DIR / "finetuned_confusion_matrix.png")
print(OUTPUT_DIR / "finetuned_predictions.csv")

print("\nClass order:", CLASSES)
