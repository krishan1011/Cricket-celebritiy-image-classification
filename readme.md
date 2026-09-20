# Cricket Celebrity Image Classification

A full-stack computer vision project that classifies Indian cricket celebrities from uploaded face images using OpenCV, wavelet-based feature extraction, and a tuned support vector machine.

This project is more than a notebook experiment: it combines a trained image classifier, a Flask API, and a browser-based UI into a working end-to-end prediction workflow.

## Project summary

Built a 9-class face recognition system for cricket celebrities and achieved measurable performance from the notebook evaluation:

- Best cross-validation score: 0.635435 (63.54%)
- Hold-out test accuracy: 0.491803 (49.18%)
- Dataset: 242 samples across 9 players
- Model: SVM selected after GridSearchCV tuning
- Feature representation: 4096-dimensional vector combining raw RGB and wavelet features

This is a strong small-dataset prototype that demonstrates end-to-end ML product development in computer vision, from preprocessing and model selection to deployment in a web app.

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
| Best CV score | 0.635435 (63.54%) |
| Final test accuracy | 0.491803 (49.18%) |

## Pipeline

Image upload → face detection → eye validation → face crop → feature extraction → SVM classification → probability output

### 1. Face detection
The app uses Haar cascades for frontal face and eye detection to isolate the subject before classification.

### 2. Feature extraction
The image is transformed into a combined feature representation:

- Raw RGB features: 32 × 32 × 3 = 3072 features
- Wavelet features: 32 × 32 = 1024 features
- Combined feature vector: 4096 features

### 3. Model selection and evaluation
The project compares multiple models using GridSearchCV and selects the best-performing classifier:

- SVM
- Random Forest
- Logistic Regression

The SVM produced the strongest validation performance and was used for final prediction.

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
├── readme.md
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

- Small dataset compared with modern deep-learning image models
- Accuracy depends on lighting, pose, face alignment, and background clutter
- Only 9 cricket players are supported
- Face detection can struggle with occlusion or low-quality images

## Future improvements

- Expand the dataset with more players and varied poses
- Add data augmentation and image balancing
- Test CNN and transfer-learning models
- Improve face detection robustness and preprocessing
- Add confidence thresholds and uncertainty handling
- Support multiple faces in a single image
- Deploy the API to a cloud platform

---

## Learning outcomes

This project demonstrates practical experience with:

- Computer vision and facial preprocessing
- Feature engineering using raw RGB and wavelet descriptors
- Model selection and evaluation with scikit-learn
- End-to-end ML deployment with Flask and a web frontend
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
