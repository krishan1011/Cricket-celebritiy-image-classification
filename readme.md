# Cricket Celebrity Image Classification

A full-stack computer vision project that identifies Indian cricket celebrities from uploaded face images using OpenCV, wavelet-based feature extraction, and a trained machine learning model.

This project combines image preprocessing, face detection, feature engineering, and a deployed Flask API into a working end-to-end classifier that predicts the player from a face image.

## Why this project

- Built a complete ML + CV workflow from image upload to prediction output
- Trained on 242 samples across 9 cricket players
- Used a handcrafted 4096-dimensional feature vector combining raw RGB and wavelet features
- Evaluated multiple classifiers with GridSearchCV before selecting the final SVM model
- Delivered a web interface and API so the model can be used outside the notebook

## Supported players

| Class | Player |
| ---: | --- |
| 0 | Gautam Gambhir |
| 1 | Hardik Pandya |
| 2 | Yashasvi Jaiswal |
| 3 | Jasprit Bumrah |
| 4 | MS Dhoni |
| 5 | Rohit Sharma |
| 6 | Sachin Tendulkar |
| 7 | Shubman Gill |
| 8 | Virat Kohli |

## Key results

| Metric | Value |
| --- | --- |
| Dataset size | 242 samples |
| Feature vector length | 4096 |
| Classes | 9 |
| Model selection method | GridSearchCV with 5-fold CV |
| Best model | SVM |
| Best CV score | 0.635435 |
| Final test accuracy | 0.491803 (49.18%) |

The notebook recorded a best cross-validation score of 63.54% and a hold-out accuracy of 49.18% on the test split. This is a strong small-dataset prototype and a solid example of end-to-end ML product development in computer vision.

---

## Pipeline

Image upload → face detection → eye validation → face crop → feature extraction → SVM classification → probability output

### 1. Face detection
The app uses Haar cascades for frontal face and eye detection.

### 2. Feature extraction
The image is converted into a combined representation:

- Raw RGB features: 32 × 32 × 3 = 3072 features
- Wavelet features: 32 × 32 = 1024 features
- Combined feature vector: 4096 features

### 3. Model evaluation
The project compares multiple algorithms using GridSearchCV:

- SVM
- Random Forest
- Logistic Regression

The SVM performed best and was selected for prediction.

---

## Tech stack

| Technology | Purpose |
| --- | --- |
| Python | Model development and backend |
| OpenCV | Face and eye detection |
| PyWavelets | Wavelet feature extraction |
| NumPy | Numerical processing |
| Pandas | Dataset preparation |
| Scikit-learn | Model selection and evaluation |
| Flask | REST API backend |
| HTML/CSS/JS | Frontend UI |
| Joblib | Model serialization |

---

## Project structure

```text
Cricket-celebritiy-image-classification/
├── model/
│   ├── opencv/
│   ├── Celebrity Image classification.ipynb
│   ├── class_dictionary.json
│   ├── requirements.txt
│   └── saved_model.pkl
├── server/
│   ├── artifacts/
│   │   ├── class_dictionary.json
│   │   └── saved_model.pkl
│   ├── opencv/
│   ├── test_images/
│   ├── b64.txt
│   ├── server.py
│   ├── util.py
│   └── wavelet.py
├── UI/
│   ├── images/
│   ├── test_images/
│   ├── app.css
│   ├── app.html
│   ├── app.js
│   ├── dropzone.min.css
│   └── dropzone.min.js
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to run

### 1. Clone the project

```bash
git clone https://github.com/krishan1011/Cricket-celebritiy-image-classification.git
cd Cricket-celebritiy-image-classification
```

### 2. Set up a virtual environment

```bash
python -m venv myenv
```

Windows:

```powershell
.\myenv\Scripts\activate
```

### 3. Install dependencies

```bash
cd model
pip install -r requirements.txt
```

### 4. Start the app

```bash
cd server
python server.py
```

Then open the frontend in the `UI` folder in a browser to upload an image and get predictions.

---

## Example output

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
      "Yashasvi_Jaiswal": 2,
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

## Limitations

- Small dataset compared to modern deep-learning image models
- Prediction quality depends on lighting, pose, face alignment, and background
- Only 9 cricket players are supported
- Face detection can struggle with occlusion or low-quality images

## Future improvements

- Expand dataset diversity and sample size
- Add data augmentation
- Test CNN and transfer-learning models
- Improve face detection robustness
- Add confidence thresholds and uncertainty handling
- Support multiple faces in a single image
- Deploy the API to a cloud platform

---

## Learning outcomes

This project demonstrates practical experience with:

- Computer vision
- Face detection and preprocessing
- Feature engineering
- Wavelet transforms
- Supervised learning
- SVM model selection and evaluation
- Full-stack ML deployment with Flask and frontend integration

---

## Author

**Krishan**

GitHub: [@krishan1011](https://github.com/krishan1011)

---

## License

This project is intended for educational and learning purposes.
