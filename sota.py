# ================================
# SOTA Deepfake Detection
# RGB + Frequency + Transformer
# ================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
from torch.fft import fft2, fftshift
import tqdm

# ================================
# CONFIG
# ================================

CSV_FILENAME = "manifest.csv"
IMAGE_COL = "image_path"
LABEL_COL = "target"

IMG_SIZE = 380
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
MODEL_SAVE_PATH = "./models/sota_deepfake_model.pth"

os.makedirs("./models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# DATA AUGMENTATION
# ================================

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# FREQUENCY TRANSFORM
# ================================

def compute_fft(img_tensor):
    """
    img_tensor: [3, H, W]
    returns: [1, H, W]
    """
    gray = img_tensor.mean(dim=0, keepdim=True)
    freq = fftshift(torch.abs(fft2(gray)))
    freq = freq / (freq.max() + 1e-8)
    return freq

# ================================
# DATASET
# ================================

class DeepfakeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, IMAGE_COL])
        label = torch.tensor(self.df.loc[idx, LABEL_COL], dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        freq = compute_fft(image)

        return image, freq, label

# ================================
# LOAD & SPLIT DATA
# ================================

df = pd.read_csv(CSV_FILENAME)

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df[LABEL_COL], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df[LABEL_COL], random_state=42
)

train_loader = DataLoader(
    DeepfakeDataset(train_df, "./", train_transforms),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    DeepfakeDataset(val_df, "./", val_transforms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ================================
# MODEL
# ================================

class SOTAFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB Backbone
        self.rgb_net = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        self.rgb_net.classifier = nn.Identity()

        # Frequency CNN
        self.freq_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        rgb_dim = 1792
        freq_dim = 64

        self.fusion = nn.Linear(rgb_dim + freq_dim, 512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, rgb, freq):
        rgb_feat = self.rgb_net(rgb)
        freq_feat = self.freq_net(freq)

        fused = torch.cat([rgb_feat, freq_feat], dim=1)
        fused = self.fusion(fused).unsqueeze(0)

        x = self.transformer(fused).squeeze(0)
        logits = self.classifier(x)
        return logits

model = SOTAFusionModel().to(device)

# ================================
# LOSS & OPTIMIZER
# ================================

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ================================
# TRAINING LOOP
# ================================

best_auc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for rgb, freq, labels in tqdm.tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    ):
        rgb = rgb.to(device)
        freq = freq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb, freq).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb.size(0)

    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for rgb, freq, labels in val_loader:
            rgb = rgb.to(device)
            freq = freq.to(device)

            logits = model(rgb, freq).squeeze(1)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    val_auc = roc_auc_score(all_labels, all_probs)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Val ROC-AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"🔥 Saved best model (AUC={best_auc:.4f})")

print("\nTraining complete.")
print(f"Best Validation ROC-AUC: {best_auc:.4f}")
print(f"Model saved at: {MODEL_SAVE_PATH}")
