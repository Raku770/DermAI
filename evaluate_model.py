import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt


# =========================
# 1. SETTINGS
# =========================

IMG_SIZE = 224
BATCH_SIZE = 8

VAL_CSV = "validation.csv"

MODEL_PATH = "models/efficientnet_b0_best.keras"

CLASS_NAMES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis"
]

CLASS_TO_INDEX = {
    "melanoma": 0,
    "nevus": 1,
    "seborrheic_keratosis": 2
}


# =========================
# 2. LOAD VALIDATION DATA
# =========================

print("=" * 60)
print("DermAI - Model Evaluation")
print("=" * 60)

val_df = pd.read_csv(VAL_CSV)

print("\nValidation images:", len(val_df))

val_labels = val_df["class"].map(CLASS_TO_INDEX).values
val_paths = val_df["image_path"].values


# =========================
# 3. IMAGE LOADING
# =========================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
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


# =========================
# 4. CREATE VALIDATION DATASET
# =========================

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=1
)

val_dataset = val_dataset.batch(BATCH_SIZE)

val_dataset = val_dataset.prefetch(1)


# =========================
# 5. LOAD TRAINED MODEL
# =========================

print("\nLoading trained model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================
# 6. MAKE PREDICTIONS
# =========================

print("\nGenerating predictions...")

predictions = model.predict(
    val_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# =========================
# 7. ACCURACY
# =========================

accuracy = accuracy_score(
    val_labels,
    predicted_labels
)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# =========================
# 8. CLASSIFICATION REPORT
# =========================

print("\nClassification Report:")
print()

report = classification_report(
    val_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)


# =========================
# 9. CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    val_labels,
    predicted_labels
)

print("Confusion Matrix:")
print(cm)


# =========================
# 10. SAVE CONFUSION MATRIX
# =========================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("DermAI - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=30
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")


print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)