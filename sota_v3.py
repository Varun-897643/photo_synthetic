# ==========================================================
# FULL SOTA Deepfake Detection (Image-only)
# Swin Transformer + Frequency Attention + Cross-Fusion
# ==========================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.fft import fft2, fftshift
import tqdm

# ================= CONFIG =================
CSV_FILENAME = "manifest.csv"
IMAGE_COL = "image_path"
LABEL_COL = "target"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 25
LR = 2e-4
MODEL_PATH = "./models/sota_full_swin_fft.pth"

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ================= FFT =================
def fft_map(x):
    gray = x.mean(dim=0, keepdim=True)
    freq = fftshift(torch.abs(fft2(gray)))
    freq = freq / (freq.max() + 1e-8)
    return freq

# ================= DATASET =================
class DeepfakeDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.loc[idx, IMAGE_COL])
        label = torch.tensor(self.df.loc[idx, LABEL_COL], dtype=torch.float32)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        freq = fft_map(img)

        return img, freq, label

# ================= LOAD DATA =================
df = pd.read_csv(CSV_FILENAME)
train_df, tmp_df = train_test_split(df, test_size=0.2, stratify=df[LABEL_COL], random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df[LABEL_COL], random_state=42)

train_loader = DataLoader(DeepfakeDataset(train_df,"./",train_tfms), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(DeepfakeDataset(val_df,"./",val_tfms), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ================= MODEL =================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_rgb, x_freq):
        # cross-attention: RGB queries, FFT keys/values
        Q = self.q(x_rgb).unsqueeze(1)      # [B,1,D]
        K = self.k(x_freq).unsqueeze(1)     # [B,1,D]
        V = self.v(x_freq).unsqueeze(1)     # [B,1,D]

        attn = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, V).squeeze(1) + x_rgb
        return out

class SotaFullDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.rgb_backbone.head = nn.Identity()
        rgb_dim = 768

        self.freq_net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()
        )

        self.fusion_attn = AttentionFusion(dim=rgb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1)
        )

    def forward(self, rgb, freq):
        rgb_feat = self.rgb_backbone(rgb)
        freq_feat = self.freq_net(freq)
        fused = self.fusion_attn(rgb_feat, freq_feat)
        logits = self.classifier(fused)
        return logits

model = SotaFullDeepfake().to(device)

# ================= TRAINING =================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

best_auc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for rgb, freq, labels in tqdm.tqdm(train_loader):
        rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(rgb,freq).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * rgb.size(0)

    # validation
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for rgb, freq, labels in val_loader:
            rgb, freq = rgb.to(device), freq.to(device)
            logits = model(rgb,freq).squeeze(1)
            probs = torch.sigmoid(logits)
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
    auc = roc_auc_score(y_true, y_score)
    total_loss /= len(train_loader.dataset)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Val ROC-AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"🔥 Saved best model (AUC={best_auc:.4f})")

print("\nTraining complete.")
print(f"Best Validation AUC: {best_auc:.4f}")
print(f"Model saved at: {MODEL_PATH}")
