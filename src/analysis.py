import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .dataset import FER2013Dataset, EMOTION_MAP
from .model import EmotionCNN
import torch.nn.functional as F

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix():
    # load saved predictions from training
    data = torch.load("models/test_predictions.pt", weights_only=False)
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[EMOTION_MAP[i] for i in range(7)]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("FER2013 Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close()
    print("Saved: models/confusion_matrix.png")

def save_example_predictions(n=16):
    csv_path = "data/fer2013.csv"
    test_ds = FER2013Dataset(csv_path, usage="PrivateTest")

    model = EmotionCNN(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    # pick random indices
    idxs = np.random.choice(len(test_ds), size=n, replace=False)

    rows = cols = int(np.sqrt(n))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    with torch.no_grad():
        for ax, idx in zip(axes.flatten(), idxs):
            img, label = test_ds[idx]  # img: [1,48,48]
            inp = img.unsqueeze(0).to(DEVICE)  # [1,1,48,48]

            logits = model(inp)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()

            img_np = img.squeeze(0).numpy()
            ax.imshow(img_np, cmap="gray")
            ax.axis("off")
            ax.set_title(
                f"T:{EMOTION_MAP[label.item()]}\nP:{EMOTION_MAP[pred]}",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig("models/example_predictions.png")
    plt.close()
    print("Saved: models/example_predictions.png")

def main():
    plot_confusion_matrix()
    save_example_predictions()

if __name__ == "__main__":
    main()
