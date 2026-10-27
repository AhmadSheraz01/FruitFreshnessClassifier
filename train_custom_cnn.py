from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset is currently in Downloads\archive\dataset\train and \test
DATA_DIR = Path(r"C:\Users\AALaptopRepairingLaB\Downloads\archive\dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 123

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Dataset folders not found. Expected:\n"
        f"{TRAIN_DIR}\n{TEST_DIR}\n"
        "Update DATA_DIR in this script if your dataset is in another location."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
if test_ds.class_names != class_names:
    raise ValueError(
        f"Train/test classes differ: {class_names} vs {test_ds.class_names}"
    )

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# A true scratch CNN: convolutional layers are trained from random initialization.
# Rescaling is inside the model so the app can pass ordinary 0-255 image pixels.
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.30),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(len(class_names), activation="softmax"),
], name="custom_cnn_scratch")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_DIR / "custom_cnn_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        mode="max",
    ),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nCustom CNN test accuracy: {accuracy * 100:.2f}%")
print(f"Saved model: {OUTPUT_DIR / 'custom_cnn_model.keras'}")
print(f"Class order: {class_names}")
print("\nNow rerun the website. Upload an image and click Analyze Fruit.")
