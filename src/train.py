# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from .model import EmotionCNN
from .utils import get_dataloaders, get_class_weights
from torch.optim.lr_scheduler import StepLR
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

def main():
    csv_path = "data/fer2013.csv"
    batch_size = 64
    epochs = 25
    lr = 1e-3
    patience = 6  # early stopping patience on val acc

    train_loader, val_loader, test_loader = get_dataloaders(csv_path, batch_size=batch_size)

    model = EmotionCNN(num_classes=7).to(DEVICE)

    # class weights to handle imbalance
    class_weights = get_class_weights(csv_path, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

    best_val_acc = 0.0
    best_epoch = 0
    os.makedirs("models", exist_ok=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}: "
              f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f}  ({elapsed:.1f}s)")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "models/best_model.pth")
            print(">> Saved new best model")

        scheduler.step()

        # early stopping
        if epoch - best_epoch >= patience:
            print(f"Stopping early at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Final: test evaluation with best model
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(f"\nTest loss={test_loss:.4f}, acc={test_acc:.4f}")

    torch.save({"y_true": y_true, "y_pred": y_pred}, "models/test_predictions.pt")

if __name__ == "__main__":
    main()
