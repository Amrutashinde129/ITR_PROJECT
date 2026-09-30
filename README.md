# Lung Cancer Detection Using CNN

## Project type
Deep Learning / Computer Vision / Medical Image Classification

## Classes
- Normal
- Benign
- Malignant

## IMPORTANT
This is an academic/research project. It is NOT a medical diagnostic system and must not be used for clinical decisions.

---

# 1. Project folder

The final project should look like:

Lung_Cancer_CNN_Ready_Project/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── prepare_dataset.py
├── requirements.txt
├── run_project.bat
│
├── dataset_raw/
│   ├── Normal cases/
│   ├── Benign cases/
│   └── Malignant cases/
│
└── dataset/
    ├── train/
    ├── validation/
    └── test/

The downloaded IQ-OTH/NCCD dataset already has the three source folders. Put/copy those three folders into `dataset_raw`.

---

# 2. Install libraries

Open Command Prompt in this folder:

python -m pip install -r requirements.txt

If TensorFlow is already working, you can simply run:

python -m pip install streamlit pillow scikit-learn matplotlib

---

# 3. Prepare dataset

Run:

python prepare_dataset.py

The script automatically:
- Finds images recursively
- Reads Normal cases, Benign cases and Malignant cases
- Splits images into 70% training, 15% validation and 15% test
- Creates the required folders
- Copies the images into the correct class folders

---

# 4. Train CNN

Run:

python train.py

The model will be saved as:

lung_cancer_cnn.keras

The class order will be saved as:

class_names.json

Training history will be saved as:

training_history.json

---

# 5. Evaluate

Run:

python evaluate.py

It generates:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Output files:
- confusion_matrix.png
- evaluation_results.txt

---

# 6. Run web application

Run:

streamlit run app.py

Upload a JPG/PNG/BMP/WebP CT image and click Predict.

---

# 7. Command-line prediction

After training:

python predict.py "C:\path\to\your\image.jpg"

---

# 8. One-click Windows option

After putting the dataset into `dataset_raw`, double-click:

run_project.bat

It will:
1. Prepare the dataset
2. Train the CNN
3. Evaluate the model
4. Start Streamlit

Training time depends on your CPU/GPU and dataset size.

---

# 9. Dataset note

Use the dataset according to its license/terms. Do not include patient-identifying information in project uploads or screenshots.

The source dataset folders expected by this project are:
- Normal cases
- Benign cases
- Malignant cases

---

# 10. Viva/project modules

1. Dataset Collection
2. Image Preprocessing
3. Dataset Splitting
4. Data Augmentation
5. CNN Model Construction
6. Model Training
7. Model Evaluation
8. Image Prediction
9. Streamlit User Interface

---

# 11. CNN architecture

Input Image (224x224x3)
        ↓
Data Augmentation
        ↓
Conv2D 32 + Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D 64 + Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D 128 + Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D 256 + Batch Normalization
        ↓
Max Pooling
        ↓
Global Average Pooling
        ↓
Dense 128
        ↓
Dropout
        ↓
Softmax Output
        ↓
Normal / Benign / Malignant
