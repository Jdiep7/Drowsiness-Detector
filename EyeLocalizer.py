import torch
import torch.nn as neural
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import timm
import os

import numpy as py
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'C:/Users/jdiep/projects/Drowsiness-Detector-1/LocalizerDataset'

class BioIDEyeDataset(Dataset):
    def __init__(self, data_dir, transform = None, normalize_coords=True):
        self.data_dir = data_dir
        self.transform = transform
        self.normalize_coords = normalize_coords
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.pgm')]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, index):
        image_filename = self.image_files[index]
        image_path = os.path.join(self.data_dir, image_filename)
        annotation_path = image_path.replace('.pgm', '.eye')

        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            coordinates = list(map(float, lines[1].split()))
        
        target_size = 128
        scale_x = target_size/orig_width
        scale_y = target_size/orig_height
        
        scaled_eyes = [
            coordinates[0] * scale_x, 
            coordinates[1] * scale_y, 
            coordinates[2] * scale_x, 
            coordinates[3] * scale_y
        ]
        
        if self.normalize_coords:
            scaled_eyes = [
                scaled_eyes[0]/target_size,
                scaled_eyes[1]/target_size,
                scaled_eyes[2]/target_size,
                scaled_eyes[3]/target_size
            ]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(scaled_eyes, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

dataset = BioIDEyeDataset(data_dir, transform)

train_size = int(0.8*len(dataset))
valid_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class EyeLocalizerRegression(neural.Module):
    def __init__(self, pretrained=True):
        super(EyeLocalizerRegression, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.features = neural.Sequential(*list(self.base_model.children())[:-1])
        self.regressor = neural.Sequential(
            neural.Linear(self.base_model.classifier.in_features, 256),
            neural.BatchNorm1d(256),
            neural.ReLU(),
            neural.Dropout(0.1),
            neural.Linear(256, 128),
            neural.BatchNorm1d(128),
            neural.ReLU(),
            neural.Dropout(0.1),
            neural.Linear(128, 4)
        )
        
    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.base_model.global_pool(x)
        x = x.view(x.size(0), -1)
        eye_coords = self.regressor(x)
        return eye_coords
    
class MixedLoss(neural.Module):
    def __init__(self, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = neural.L1Loss()
        self.mse_loss = neural.MSELoss()
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * mse

num_epochs = 80

model = EyeLocalizerRegression(pretrained=True)
model = model.to(device)

criterion = MixedLoss(alpha=0.7)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.3
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Training Loop"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss/len(dataloader.dataset)
    return epoch_loss

def validate(mode, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation Loop"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()*images.size(0)
            
    val_loss = running_loss / len(dataloader.dataset)
    
    return val_loss

best_val_loss = float('inf')
best_model_path = 'best_eye_localizer_model.pth'

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, valid_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, ")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved with Val Loss: {val_loss:.4f}")
    
    print("-" * 60)

print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
model.load_state_dict(torch.load(best_model_path))
    
    
    
       