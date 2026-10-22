import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# =========================
# 1. SETTINGS
# =========================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5

TRAIN_CSV = "train.csv"
VAL_CSV = "validation.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# Our 3 classes
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


print("=" * 60)
print("DermAI - EfficientNet-B0 Training")
print("=" * 60)


# =========================
# 2. LOAD CSV FILES
# =========================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["class"].value_counts())

print("\nValidation class distribution:")
print(val_df["class"].value_counts())


# =========================
# 3. CONVERT CLASS NAMES
# =========================

train_labels = train_df["class"].map(CLASS_TO_INDEX).values
val_labels = val_df["class"].map(CLASS_TO_INDEX).values

train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values


# =========================
# 4. IMAGE LOADING FUNCTION
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
# 5. CREATE TF.DATA DATASETS
# =========================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_df),
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=1
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(1)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=1
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(1)


# =========================
# 6. DATA AUGMENTATION
# =========================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# =========================
# 7. LOAD EFFICIENTNET-B0
# =========================

print("\nLoading EfficientNet-B0...")

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


# =========================
# 8. BUILD MODEL
# =========================

inputs = keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    3,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# =========================
# 9. COMPILE MODEL
# =========================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

model.summary()


# =========================
# 10. CLASS WEIGHTS
# =========================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_labels
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1],
    2: class_weights_array[2]
}

print("\nClass weights:")
print(class_weights)


# =========================
# 11. CALLBACKS
# =========================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_best.keras"
)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
]


# =========================
# 12. TRAIN MODEL
# =========================

print("\nStarting training...")
print("This may take some time because your system is using CPU.")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================
# 13. SAVE FINAL MODEL
# =========================

final_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_skin_lesion.keras"
)

model.save(final_model_path)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nFinal model saved at:")
print(final_model_path)

print("\nBest model saved at:")
print(checkpoint_path)

print("\nTraining history:")
print(history.history)