# Emotion Recognition (FER2013)

A clean, minimal PyTorch project that trains a CNN to classify **7 emotions** from the FER2013 dataset, with a **webcam demo** included.

## What this project includes

* Custom FER2013 dataset loader (from `.csv`)
* Lightweight CNN model (3 conv blocks)
* Training pipeline with:

  * data augmentation
  * soft class weighting
  * LR scheduler + early stopping
* Evaluation scripts (confusion matrix, sample predictions)
* Real-time webcam emotion detection using OpenCV

---

## How to use

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Add dataset

Download fer2013.csv from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
Then place it in:
```
data/fer2013.csv
```

*(Dataset is excluded from GitHub.)*

### 3. Train the model

```bash
python -m src.train
```

Trained weights + analysis images are saved into `models/`.

### 4. Run analysis

```bash
python -m src.analysis
```

Outputs:

* `confusion_matrix.png`
* `example_predictions.png`

### 5. Webcam demo

```bash
python -m src.webcam_demo
```

Press **q** to exit.

---

## Project structure

```
emotion-recognition/
├── data/          # dataset (ignored)
├── models/        # trained models + images (ignored)
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── analysis.py
│   ├── utils.py
│   └── webcam_demo.py
├── README.md
└── requirements.txt
```

---

## Notes

* FER2013 splits (Training/PublicTest/PrivateTest) are respected.
* Webcam demo uses Haar cascades for face detection.
* Repo contains **only code**, not weights or data.

---

