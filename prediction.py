import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras


# =========================
# 1. SETTINGS
# =========================

IMG_SIZE = 224

MODEL_PATH = "models/efficientnet_b0_best.keras"

CLASS_NAMES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis"
]


# =========================
# 2. LOAD MODEL
# =========================

print("Loading DermAI model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# =========================
# 3. CHECK IMAGE PATH
# =========================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python prediction.py path_to_image.jpg")
    sys.exit()


image_path = sys.argv[1]


# =========================
# 4. LOAD IMAGE
# =========================

print("\nLoading image...")

image = tf.io.read_file(image_path)

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

# Add batch dimension
image = tf.expand_dims(
    image,
    axis=0
)


# =========================
# 5. PREDICT
# =========================

print("Running prediction...")

prediction = model.predict(
    image,
    verbose=0
)

predicted_index = np.argmax(
    prediction[0]
)

predicted_class = CLASS_NAMES[
    predicted_index
]

confidence = prediction[0][
    predicted_index
] * 100


# =========================
# 6. DISPLAY RESULT
# =========================

print("\n" + "=" * 50)
print("DERMAI PREDICTION")
print("=" * 50)

print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("=" * 50)

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = prediction[0][i] * 100

    print(
        f"{class_name}: {probability:.2f}%"
    )

print("\nIMPORTANT:")
print("This is a preliminary AI classification result.")
print("It is NOT a medical diagnosis.")