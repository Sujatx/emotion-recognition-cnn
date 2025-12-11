# src/utils.py
from torch.utils.data import DataLoader
from .dataset import FER2013Dataset
import pandas as pd
import torch

def get_dataloaders(csv_path, batch_size=64):
    train_ds = FER2013Dataset(csv_path, usage="Training")
    val_ds   = FER2013Dataset(csv_path, usage="PublicTest")
    test_ds  = FER2013Dataset(csv_path, usage="PrivateTest")

    # num_workers=0 on Windows is safe
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def get_class_weights(csv_path, device):
    """
    Compute softer class weights (avoid huge weights for tiny classes).
    Returns a torch tensor on device.
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Usage"] == "Training"]
    counts = train_df["emotion"].value_counts().sort_index()  # 0..6
    counts = counts.to_numpy().astype(float)
    counts[counts == 0] = 1.0

    # SOFTER inverse-frequency: use sqrt to reduce extremes
    weights = 1.0 / (counts ** 0.5)

    # normalize so mean weight = 1 (keeps scale sane)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)

