# src/dataset.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# Map label -> emotion name
EMOTION_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

class FER2013Dataset(Dataset):
    def __init__(self, csv_path, usage="Training", transform=None):
        """
        usage: 'Training', 'PublicTest', 'PrivateTest'
        """
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["Usage"] == usage].reset_index(drop=True)
        self.transform = transform
        self.usage = usage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # label
        emotion = int(row["emotion"])

        # pixels -> numpy array -> 48x48
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)           # (48,48) grayscale
        img = img.astype(np.float32) / 255.0   # normalize to [0,1]
        img = np.expand_dims(img, axis=0)      # (1,48,48)
        img = torch.from_numpy(img)            # tensor float32

        # --- simple, safe augmentations for training only ---
        if self.usage == "Training":
            # horizontal flip 50%
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])  # flip width dimension

            # small contrast jitter (multiply by scalar)
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                img = torch.clamp(img * factor, 0.0, 1.0)

            # tiny gaussian noise
            if random.random() < 0.3:
                noise = torch.randn_like(img) * 0.03
                img = torch.clamp(img + noise, 0.0, 1.0)

        # allow an optional transform (kept for compatibility)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(emotion, dtype=torch.long)
        return img, label
