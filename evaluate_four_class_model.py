
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Class order must match training
CLASSES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis",
    "non_lesional_skin"
]

IMG_SIZE = 224
BATCH_SIZE = 8

# Load validation data
val_df = pd.read_csv("validation_four_class.csv")

# Load best trained model
model = tf.keras.models.load_model(
    "models/efficientnet_b0_four_class_best.keras"
)

# Load and preprocess images
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label

class_map = {name: i for i, name in enumerate(CLASSES)}

labels = val_df["class"].map(class_map).to_numpy()

dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["image_path"].astype(str).to_numpy(), labels)
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE)

# Generate predictions
print("Evaluating four-class model...")

predictions = model.predict(dataset, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Overall accuracy
accuracy = accuracy_score(labels, predicted_labels)

print("\nOverall validation accuracy:", round(accuracy * 100, 2), "%")

# Class-wise performance
print("\nCLASSIFICATION REPORT:")
print(classification_report(
    labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    target_names=CLASSES,
    zero_division=0
))

# Confusion matrix
print("\nCONFUSION MATRIX:")
print(confusion_matrix(
    labels,
    predicted_labels,
    labels=[0, 1, 2, 3]
))

print("\nEvaluation completed!")
