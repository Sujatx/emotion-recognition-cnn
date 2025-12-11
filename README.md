# Emotion Recognition from Faces (FER2013)

A compact, end-to-end PyTorch project that trains a convolutional neural network to classify seven facial expressions from the FER2013 dataset: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

This repository contains a reproducible training pipeline, analysis utilities (confusion matrix and example predictions), and a simple real-time webcam demo for inference.

---

## Features

* Custom `Dataset` / `DataLoader` for the FER2013 CSV
* Lightweight CNN implemented in PyTorch (easy to read and extend)
* Data augmentation (safe transforms) and soft class-weighting for imbalance
* LR scheduler and early stopping to stabilize training
* Scripts to train, evaluate, and visualize results (confusion matrix, example predictions)
* Real-time webcam demo using OpenCV for live inference
* Optional Colab recipe for fine-tuning a ResNet backbone (recommended for higher accuracy)

---

## Quick start

> Clone the repository, create a Python virtual environment, install dependencies, place `fer2013.csv` in `data/`, and run the training script.

### 1. Clone

```bash
git clone https://github.com/your-username/emotion-recognition-cnn.git
cd emotion-recognition-cnn
```

### 2. Prepare environment (Windows PowerShell example)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` should include at minimum:

```
torch
torchvision
pandas
numpy
matplotlib
scikit-learn
opencv-python
```

### 3. Download FER2013

FER2013 is hosted on Kaggle. Download `fer2013.csv` from:

* *Challenges in Representation Learning: Facial Expression Recognition Challenge (FER2013)* on Kaggle

**Do not upload the dataset to GitHub.** Place the CSV in:

```
emotion-recognition-cnn/data/fer2013.csv
```

### 4. Train (local)

```powershell
python -m src.train
```

This runs training with augmentation, soft class-weights, LR scheduler and early stopping. Models and artifacts are saved to `models/`.

### 5. Analyze results

After training run:

```powershell
python -m src.analysis
```

This generates `models/confusion_matrix.png` and `models/example_predictions.png`.

### 6. Webcam demo

Make sure your venv is active and OpenCV installed, then run:

```powershell
python -m src.webcam_demo
```

The demo performs face detection, crops and preprocesses faces, and shows live predictions. Press `q` to quit.

---

## Project structure

```
emotion-recognition-cnn/
├─ data/                # dataset (not included)
├─ models/              # saved checkpoints + generated images
├─ notebooks/           # optional analysis / Colab notes
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  ├─ utils.py
│  ├─ analysis.py
│  └─ webcam_demo.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Reproducibility notes

* FER2013 includes `Usage` column (Training / PublicTest / PrivateTest). The code uses these splits rather than re-splitting the CSV.
* Training configuration (batch size, learning rate, scheduler, number of epochs, augmentation) can be found and adjusted in `src/train.py`.
* A previous baseline run produced ~56% test accuracy on the small CNN after applying soft class weights and augmentation. Fine-tuning a ResNet backbone on GPU typically yields higher accuracy (see `notebooks/` for a Colab recipe).

---

## Tips & Troubleshooting

* If OpenCV windows do not appear on Windows, try running from a regular PowerShell (not the VS Code integrated terminal) or update `opencv-python`.
* If training is slow locally, use the provided Colab recipe (in `notebooks/`) to run on GPU.
* If `torch.load` raises security-related unpickling errors, use `torch.load(path, weights_only=False)` for non-weight checkpoint files you trust.

---

## Extensions (ideas)

* Fine-tune a pretrained ResNet18/ResNet34 backbone (Colab recommended)
* Add face alignment (MTCNN / dlib) before inference for better live performance
* Convert model to ONNX/TFLite for deployment on edge devices
* Build a small Streamlit app for demoing predictions in-browser

---

## License

This repository uses the MIT License — see `LICENSE` (add to repo if desired).

---

## Acknowledgements

* FER2013 dataset (Kaggle) — used under the dataset's terms
* PyTorch, torchvision, OpenCV and scikit-learn

---
