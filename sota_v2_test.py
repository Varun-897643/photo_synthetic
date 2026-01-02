# ================= TESTING SCRIPT =================

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from your_model_file import DeepfakeDataset, SwinFFTDetector, fft_map  # adjust import

# ================= CONFIG =================
CSV_FILENAME = "manifest.csv"  # same CSV
IMAGE_COL = "image_path"
LABEL_COL = "target"
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = "./models/swin_fft_deepfake.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DATA =================
df = pd.read_csv(CSV_FILENAME)
# Use the same test split as during training
# Assuming you saved test_df, else split again
from sklearn.model_selection import train_test_split
_, tmp_df = train_test_split(df, test_size=0.2, stratify=df[LABEL_COL], random_state=42)
_, test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df[LABEL_COL], random_state=42)

test_loader = DataLoader(
    DeepfakeDataset(test_df, "./", test_tfms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# ================= MODEL =================
model = SwinFFTDetector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================= INFERENCE =================
y_true, y_score, y_pred = [], [], []

with torch.no_grad():
    for rgb, freq, labels in test_loader:
        rgb, freq = rgb.to(device), freq.to(device)
        logits = model(rgb, freq).squeeze(1)
        probs = torch.sigmoid(logits)

        y_true.extend(labels.numpy())
        y_score.extend(probs.cpu().numpy())
        y_pred.extend((probs.cpu().numpy() > 0.5).astype(int))

# ================= METRICS =================
auc = roc_auc_score(y_true, y_score)
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Test ROC-AUC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1-score: {f1:.4f}")

# Optional: Save predictions
results_df = pd.DataFrame({
    "image_path": test_df[IMAGE_COL].values,
    "true_label": y_true,
    "pred_prob": y_score,
    "pred_label": y_pred
})
results_df.to_csv("test_predictions.csv", index=False)
print("✅ Predictions saved to test_predictions.csv")
