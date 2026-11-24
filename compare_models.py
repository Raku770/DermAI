
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 8

CLASSES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis",
    "non_lesional_skin"
]

CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

# Load the original 400-image validation set
df = pd.read_csv("validation.csv")

labels = df["class"].map(CLASS_MAP).to_numpy()

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label

dataset = tf.data.Dataset.from_tensor_slices(
    (df["image_path"].astype(str).to_numpy(), labels)
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE)

# Load four-class model
model = tf.keras.models.load_model(
    "models/efficientnet_b0_four_class_best.keras"
)

print("Testing four-class model on original 400 images...")

probabilities = model.predict(dataset, verbose=1)
predictions = np.argmax(probabilities, axis=1)

print("\nAccuracy on the same 400 images:")
print(round(accuracy_score(labels, predictions) * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    labels,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=CLASSES,
    zero_division=0
))

print("\nNumber predicted as non-lesional skin:",
      int(np.sum(predictions == 3)))

print("\nComparison test completed!")
