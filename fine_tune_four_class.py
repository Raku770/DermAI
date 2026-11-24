
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------
# 1. Configuration
# ---------------------------------

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.00001
UNFREEZE_LAST_LAYERS = 30

TRAIN_CSV = "train_four_class.csv"
VAL_CSV = "validation_four_class.csv"

MODEL_PATH = Path("models/efficientnet_b0_four_class_best.keras")

BEST_MODEL_PATH = Path(
    "models/efficientnet_b0_four_class_finetuned_best.keras"
)

FINAL_MODEL_PATH = Path(
    "models/efficientnet_b0_four_class_finetuned.keras"
)

CLASSES = [
    "melanoma",
    "nevus",
    "seborrheic_keratosis",
    "non_lesional_skin"
]

CLASS_MAP = {name: index for index, name in enumerate(CLASSES)}

# ---------------------------------
# 2. Verify files
# ---------------------------------

assert MODEL_PATH.is_file(), "Four-class model not found!"
assert Path(TRAIN_CSV).is_file(), "Training CSV not found!"
assert Path(VAL_CSV).is_file(), "Validation CSV not found!"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))

assert set(train_df["class"]) == set(CLASSES)
assert set(val_df["class"]) == set(CLASSES)

assert train_df["image_path"].apply(
    lambda p: Path(p).is_file()
).all()

assert val_df["image_path"].apply(
    lambda p: Path(p).is_file()
).all()

# ---------------------------------
# 3. Prepare labels and weights
# ---------------------------------

train_labels = train_df["class"].map(CLASS_MAP).to_numpy()
val_labels = val_df["class"].map(CLASS_MAP).to_numpy()

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=train_labels
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:", class_weights)

# ---------------------------------
# 4. Image loading
# ---------------------------------

def load_image(path, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image, [IMG_SIZE, IMG_SIZE]
    )
    image = tf.cast(image, tf.float32)

    return image, label


def create_dataset(df, labels, training=False):

    paths = df["image_path"].astype(str).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels.astype(np.int32))
    )

    if training:
        dataset = dataset.shuffle(
            len(df),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_df, train_labels, training=True
)

val_ds = create_dataset(
    val_df, val_labels
)

# ---------------------------------
# 5. Load existing trained model
# ---------------------------------

print("\nLoading existing four-class model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Existing model loaded successfully!")

print("\nModel layers:")
for layer in model.layers:
    print(layer.name, type(layer).__name__)

# ---------------------------------
# 6. Find EfficientNet backbone
# ---------------------------------

base_model = next(
    (
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "efficientnet" in layer.name.lower()
    ),
    None
)

if base_model is None:
    raise ValueError(
        "EfficientNet backbone not found. "
        "Check the model layer names before training."
    )

print("\nEfficientNet backbone:", base_model.name)

# ---------------------------------
# 7. Fine-tune last 30 layers
# ---------------------------------

base_model.trainable = True

# Freeze all backbone layers first
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze only the last selected layers
for layer in base_model.layers[-UNFREEZE_LAST_LAYERS:]:

    # Keep BatchNormalization layers frozen
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

print("\nTrainable backbone layers:")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

# ---------------------------------
# 8. Compile model
# ---------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------
# 9. Callbacks
# ---------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(BEST_MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ---------------------------------
# 10. Fine-tuning
# ---------------------------------

print("\nStarting four-class fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ---------------------------------
# 11. Save final model
# ---------------------------------

model.save(str(FINAL_MODEL_PATH))

print("\nFine-tuning completed!")

print("Best model:", BEST_MODEL_PATH)
print("Final model:", FINAL_MODEL_PATH)

# ---------------------------------
# 12. Display final metrics
# ---------------------------------

best_epoch = np.argmin(history.history["val_loss"])

print("\nBest epoch:", best_epoch + 1)

print(
    "Training accuracy:",
    history.history["accuracy"][best_epoch]
)

print(
    "Validation accuracy:",
    history.history["val_accuracy"][best_epoch]
)

print(
    "Validation loss:",
    history.history["val_loss"][best_epoch]
)

print("\nClass order:", CLASSES)
