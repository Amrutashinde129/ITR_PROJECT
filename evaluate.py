import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

MODEL_PATH = "lung_cancer_cnn.keras"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run 'python train.py' first.")

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError("Test dataset not found. Run 'python prepare_dataset.py' first.")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

class_names = test_ds.class_names
model = tf.keras.models.load_model(MODEL_PATH)

loss, keras_accuracy = model.evaluate(test_ds, verbose=1)

y_true = []
for _, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

probabilities = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print("\n================ EVALUATION ================")
print(f"Keras test accuracy : {keras_accuracy * 100:.2f}%")
print(f"Accuracy            : {accuracy * 100:.2f}%")
print(f"Weighted precision  : {precision * 100:.2f}%")
print(f"Weighted recall     : {recall * 100:.2f}%")
print(f"Weighted F1-score   : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot(values_format="d")
plt.title("Lung Cancer CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Keras test accuracy: {keras_accuracy * 100:.2f}%\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Weighted precision: {precision * 100:.2f}%\n")
    f.write(f"Weighted recall: {recall * 100:.2f}%\n")
    f.write(f"Weighted F1-score: {f1 * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

print("\nSaved: confusion_matrix.png")
print("Saved: evaluation_results.txt")
