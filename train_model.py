import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
# -----------------------------
# Configuration
# -----------------------------
TRAIN_DIR = "train"
MODEL_DIR = "model"
IMAGE_SIZE = 28
NUM_CLASSES = 5
EPOCHS = 10
os.makedirs(MODEL_DIR, exist_ok=True)
# -----------------------------
# Load training images
# -----------------------------
images = []
labels = []
categories = ["A", "B", "C", "D", "E"]
label_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}
for category in categories:
    folder = os.path.join(TRAIN_DIR, category)
    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            image = cv2.imread(filepath)
            if image is None:
                continue
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            # Convert image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(label_map[category])
        except Exception as e:
            print("Error:", filepath, e)
# -----------------------------
# Check dataset
# -----------------------------
if len(images) == 0:
    print()
    print("ERROR: No training images found.")
    print("Please add images inside:")
    print("train/A")
    print("train/B")
    print("train/C")
    print("train/D")
    print("train/E")
    exit()
print("Total images:", len(images))
# -----------------------------
# Prepare data
# -----------------------------
X = np.array(images, dtype=np.float32)
y = np.array(labels)
X = X.reshape(-1, 28, 28, 1)
# Normalize pixel values
X = X / 255.0
# Convert labels to categorical
y = to_categorical(y, NUM_CLASSES)
print("X shape:", X.shape)
print("Y shape:", y.shape)
# -----------------------------
# Build CNN / LeNet-style model
# -----------------------------
model = Sequential()
model.add(
    Conv2D(
        28,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    )
)
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
model.add(Flatten())
model.add(
    Dense(128, activation="relu")
)
model.add(
    Dropout(0.2)
)
model.add(
    Dense(NUM_CLASSES, activation="softmax")
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print()
print("Model Summary:")
model.summary()
# -----------------------------
# Train model
# -----------------------------
print()
print("Training started...")
model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)
# -----------------------------
# Save model architecture
# -----------------------------
model_json = model.to_json()
json_path = os.path.join(
    MODEL_DIR,
    "model.json"
)
with open(json_path, "w") as json_file:
    json_file.write(model_json)
# -----------------------------
# Save model weights
# -----------------------------
weights_path = os.path.join(
    MODEL_DIR,
    "model_weights.h5"
)
model.save_weights(weights_path)
print()
print("====================================")
print("MODEL TRAINING COMPLETED")
print("====================================")
print("Model architecture:")
print(json_path)
print("Model weights:")
print(weights_path)
print()
print("You can now use the generated model for prediction.")
