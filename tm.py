import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import tqdm

CSV_FILENAME = 'manifest.csv'
IMAGE_COL = 'image_path'
LABEL_COL = 'target'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10 
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = './models/vgg16_fake_detection_pytorch.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


df = pd.read_csv(CSV_FILENAME)

print("Total Samples:", len(df))

train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df[LABEL_COL]
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df[LABEL_COL]
)

print(f"Training Samples: {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")
print(f"Test Samples: {len(test_df)}")


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ManifestDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, self.dataframe.columns.get_loc(IMAGE_COL)])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, self.dataframe.columns.get_loc(LABEL_COL)]
        label = torch.tensor(label, dtype=torch.float32) # Convert to tensor
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = ManifestDataset(dataframe=train_df, root_dir='./', transform=train_transforms)
val_dataset = ManifestDataset(dataframe=val_df, root_dir='./', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("PyTorch Datasets and DataLoaders created successfully.")



model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features 
model.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),      
    nn.Sigmoid()            
)

model = model.to(device)

criterion = nn.BCELoss() 
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

print("VGG-16 Model modified and moved to device.")



best_val_accuracy = 0.0

print("\nStarting Model Training on PyTorch...")

for epoch in range(NUM_EPOCHS):
    model.train() 
    running_loss = 0.0
    
    for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze(1)
            predicted = (outputs > 0.5).float() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct / total

    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'  Train Loss: {epoch_loss:.4f}')
    print(f'  Validation Accuracy: {epoch_accuracy:.4f}')
    
    if epoch_accuracy > best_val_accuracy:
        best_val_accuracy = epoch_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Saved best model weights to: {MODEL_SAVE_PATH} (Accuracy: {best_val_accuracy:.4f})")

print("\nTraining Complete!")

