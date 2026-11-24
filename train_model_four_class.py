
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# 1. Configuration
# -------------------------------

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5

TRAIN_CSV = "train_four_class.csv"
VAL_CSV = "validation_four_class.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis",
    "non_lesional_skin"
]

CLASS_MAP = {name: index for index, name in enumerate(CLASSES)}

# -------------------------------
# 2. Load CSV files
# -------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("\nTraining distribution:")
print(train_df["class"].value_counts())

# Check classes and image paths
assert set(train_df["class"]) == set(CLASSES)
assert set(val_df["class"]) == set(CLASSES)
assert train_df["image_path"].apply(lambda p: Path(p).is_file()).all()
assert val_df["image_path"].apply(lambda p: Path(p).is_file()).all()

# Convert class names to numeric labels
train_labels = train_df["class"].map(CLASS_MAP).to_numpy()
val_labels = val_df["class"].map(CLASS_MAP).to_numpy()

# -------------------------------
# 3. Calculate class weights
# -------------------------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=train_labels
)

class_weights = {
    i: float(weight) for i, weight in enumerate(weights)
}

print("\nClass weights:", class_weights)

# -------------------------------
# 4. Image loading
# -------------------------------

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label

def create_dataset(df, labels, training=False):
    paths = df["image_path"].astype(str).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels.astype(np.int32))
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = create_dataset(train_df, train_labels, training=True)
val_ds = create_dataset(val_df, val_labels)

# -------------------------------
# 5. Data augmentation
# -------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# -------------------------------
# 6. EfficientNetB0 model
# -------------------------------

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# -------------------------------
# 7. Compile model
# -------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 8. Save best model separately
# -------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(MODEL_DIR / "efficientnet_b0_four_class_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# -------------------------------
# 9. Train model
# -------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# -------------------------------
# 10. Save final model
# -------------------------------

model.save(
    str(MODEL_DIR / "efficientnet_b0_four_class_final.keras")
)

print("\nFour-class training completed!")
print("Best model:", MODEL_DIR / "efficientnet_b0_four_class_best.keras")
print("Final model:", MODEL_DIR / "efficientnet_b0_four_class_final.keras")
print("Class order:", CLASSES)
