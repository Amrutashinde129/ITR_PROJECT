import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "lung_cancer_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

if len(sys.argv) < 2:
    print("Usage: python predict.py path_to_image.jpg")
    raise SystemExit(1)

image_path = sys.argv[1]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run python train.py first.")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
array = np.asarray(image, dtype=np.float32) / 255.0
array = np.expand_dims(array, axis=0)

probabilities = model.predict(array, verbose=0)[0]
index = int(np.argmax(probabilities))

print(f"\nPrediction: {class_names[index]}")
print(f"Confidence: {probabilities[index] * 100:.2f}%")

print("\nAll probabilities:")
for name, probability in zip(class_names, probabilities):
    print(f"{name}: {probability * 100:.2f}%")

print("\nEducational/research use only; not a medical diagnosis.")
