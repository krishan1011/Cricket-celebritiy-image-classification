# Cricket Celebrity Image Classification

An end-to-end **Machine Learning + Computer Vision** project that identifies famous Indian cricket players from images using **OpenCV, Wavelet Transform, Scikit-learn, Flask, HTML, CSS, and JavaScript**.

The application allows a user to upload an image through a web interface, detects the face, extracts image features, and predicts which cricket celebrity appears in the image along with the prediction probabilities.

---

##  Project Overview

This project implements a complete image-classification pipeline:

**Image Upload → Face Detection → Face Cropping → Feature Extraction → ML Classification → Prediction → Web UI**

The system is trained to recognize the following **9 cricket players**:

| Class | Cricket Player   |
| ----: | ---------------- |
|     0 | Gautam Gambhir   |
|     1 | Hardik Pandya    |
|     2 | Yashasvi Jaiswal |
|     3 | Jasprit Bumrah   |
|     4 | MS Dhoni         |
|     5 | Rohit Sharma     |
|     6 | Sachin Tendulkar |
|     7 | Shubman Gill     |
|     8 | Virat Kohli      |

The class mapping is stored in `class_dictionary.json`.

---

## Features

* Upload a cricket player's image through a web interface
* Detect faces using OpenCV Haar Cascades
* Validate detected faces using eye detection
* Automatically crop the detected face
* Extract Wavelet Transform features
* Generate a **4096-dimensional feature vector**
* Classify players using a Scikit-learn ML model
* Display prediction probabilities for all supported players
* Flask REST API for model inference
* Interactive HTML/CSS/JavaScript frontend
* Pre-trained model included for inference

---

## Machine Learning Pipeline

### 1. Face Detection

The project uses OpenCV's Haar Cascade classifiers:

* `haarcascade_frontalface_default.xml`
* `haarcascade_eye.xml`

An image is considered suitable for classification when a detected face contains at least two detected eyes.

### 2. Image Cropping

Detected faces are cropped from the original images and stored in class-specific directories.

### 3. Feature Extraction

Two types of features are combined:

#### Raw Image Features

The cropped face is resized to:

```text
32 × 32 × 3
```

which produces:

```text
3072 features
```

#### Wavelet Features

A **Discrete Wavelet Transform (DWT)** is applied using the `db1` wavelet at level 5.

The resulting image is resized to:

```text
32 × 32
```

giving:

```text
1024 features
```

### 4. Combined Feature Vector

The raw RGB features and wavelet features are concatenated:

```text
3072 + 1024 = 4096 features
```

The notebook confirms that the processed dataset contains **242 samples with 4096 features per sample**.

---

## Model Selection

The project compares multiple machine-learning algorithms using `GridSearchCV` with 5-fold cross-validation:

* Support Vector Machine
* Random Forest
* Logistic Regression

The trained SVM estimator is then used by the application for prediction.

> **Note:** These results are based on the dataset and experimental split contained in the notebook. Model performance can vary depending on the dataset, image quality, train/test split, and environment.

---

## 🏗️ Project Architecture

```text
                         ┌──────────────────┐
                         │    User Image    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Web UI        │
                         │ HTML/CSS/JS      │
                         └────────┬─────────┘
                                  │
                                  │ Base64 Image
                                  ▼
                         ┌──────────────────┐
                         │   Flask Server   │
                         │     Port 5000    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Face + Eye       │
                         │ Detection        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Face Cropping    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Feature          │
                         │ Extraction       │
                         │ RGB + Wavelet    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Trained SVM      │
                         │ Classifier       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Player +         │
                         │ Probabilities    │
                         └──────────────────┘
```

---

## Project Structure

```text
Cricket-celebritiy-image-classification/
│
├── model/
│   ├── opencv/
│   ├── Celebrity Image classification.ipynb
│   ├── class_dictionary.json
│   ├── requirements.txt
│   └── saved_model.pkl
│
├── server/
│   ├── artifacts/
│   │   ├── class_dictionary.json
│   │   └── saved_model.pkl
│   │
│   ├── opencv/
│   ├── test_images/
│   ├── b64.txt
│   ├── server.py
│   ├── util.py
│   └── wavelet.py
│
├── UI/
│   ├── images/
│   ├── test_images/
│   ├── app.css
│   ├── app.html
│   ├── app.js
│   ├── dropzone.min.css
│   └── dropzone.min.js
│
├── .gitignore
└── README.md
```

The repository is organized into separate **model**, **server**, and **UI** components.

---

## Technologies Used

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| **Python**       | Machine learning and backend |
| **OpenCV**       | Face and eye detection       |
| **PyWavelets**   | Wavelet feature extraction   |
| **NumPy**        | Numerical operations         |
| **Pandas**       | Data handling                |
| **Scikit-learn** | ML models and GridSearchCV   |
| **Flask**        | Backend REST API             |
| **HTML/CSS**     | Frontend                     |
| **JavaScript**   | Frontend interaction         |
| **Dropzone.js**  | Image upload interface       |
| **Joblib**       | Model serialization          |

The project's requirements include OpenCV, PyWavelets, Flask, NumPy, Pandas, Scikit-learn, SciPy, and related packages.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/krishan1011/Cricket-celebritiy-image-classification.git
```

```bash
cd Cricket-celebritiy-image-classification
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv myenv
```

Activate it:

```powershell
.\myenv\Scripts\activate
```

If PowerShell blocks script execution, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

and then:

```powershell
.\myenv\Scripts\activate
```
---

## 3. Install Dependencies

Move into the model directory:

```bash
cd model
```

Install the required packages:

```bash
pip install -r requirements.txt
```
---

# Running the Application

## Step 1 — Start the Flask Backend

Open a terminal and navigate to:

## Step 2 — Open the Frontend

Navigate to the `UI` directory:

---

# How Prediction Works

When an image is uploaded:

```text
1. User uploads image
          ↓
2. JavaScript converts image to Base64
          ↓
3. Base64 data sent to Flask API
          ↓
4. OpenCV detects faces
          ↓
5. OpenCV detects eyes
          ↓
6. Valid face is cropped
          ↓
7. Image resized to 32 × 32
          ↓
8. Wavelet features extracted
          ↓
9. RGB + Wavelet features combined
          ↓
10. Feature vector passed to SVM
          ↓
11. Class probabilities generated
          ↓
12. Highest probability player displayed
```

The backend performs the same preprocessing during inference as was used during training, including resizing, wavelet transformation, and feature concatenation.

---

# Prediction Output

The backend returns information similar to:

```json
[
    {
        "class": "Virat_Kohli",
        "class_probability": [
            0.12,
            0.31,
            0.04,
            0.25,
            0.18,
            0.02,
            0.03,
            98.75,
            0.50
        ],
        "class_dictionary": {
            "Gautam_Gambhir": 0,
            "Hardik_Pandya": 1,
            "jaiswal": 2,
            "Jasprit_Bumrah": 3,
            "MS_Dhoni": 4,
            "Rohit_Sharma": 5,
            "Sachin_Tendulkar": 6,
            "Shubman_Gill": 7,
            "Virat_Kohli": 8
        }
    }
]
```

---


# Limitations

The current implementation has several limitations:

* The dataset is relatively small.
* Prediction performance depends heavily on image quality.
* The model is trained only on the supported nine players.
* Images without clearly detectable faces/eyes may not be classified.
* Images with multiple faces can produce ambiguous results.
* Lighting, pose, occlusion, and background variations can affect predictions.
* The current model is a traditional ML pipeline rather than a deep CNN/transfer-learning model.

---

# Future Improvements

Potential improvements include:

* [ ] Increase the size and diversity of the dataset
* [ ] Add more cricket players
* [ ] Use data augmentation
* [ ] Experiment with CNN architectures
* [ ] Compare against transfer-learning models such as ResNet, MobileNet, or EfficientNet
* [ ] Improve face detection using modern detectors
* [ ] Add confidence thresholds for uncertain predictions
* [ ] Support multiple faces in a single image
* [ ] Deploy the Flask API to a cloud platform
* [ ] Containerize the application using Docker
* [ ] Add automated testing
* [ ] Add a production-ready REST API
* [ ] Improve frontend responsiveness and UX

---

# Learning Outcomes

This project demonstrates practical experience with:

* Computer Vision
* Image preprocessing
* Face detection
* Feature engineering
* Wavelet transforms
* Supervised Machine Learning
* SVM classification
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* Model serialization
* REST API development
* Frontend-backend integration

---

# Author

**Krishan**

GitHub: [@krishan1011](https://github.com/krishan1011)

---

## 📄 License

This project is intended for educational and learning purposes.
