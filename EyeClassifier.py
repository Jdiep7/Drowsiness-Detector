import torch
import torch.nn as neural
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as py
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
    
class SimpleEyeClassifier(neural.Module):
    def __init__(self, num_classes=2):
        super(SimpleEyeClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0',pretrained=True)
        self.features = neural.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = neural.Linear(self.base_model.classifier.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

data_dir='C:/Users/jdiep/projects/Drowsiness-Detector/ClassifierDataset'
dataset = EyeDataSet(data_dir, transform)

train_size = int(0.8*len(dataset))
valid_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SimpleEyeClassifier(num_classes=2).to(device)
criterion = neural.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct  = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training Loop"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss/ len(train_loader)
    train_accuracy = 100* correct/total
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation Loop"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
    val_loss_avg = val_loss / len(valid_loader)
    val_accuracy = 100 * val_correct / val_total
        
    print(f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
        f"Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
torch.save(model.state_dict(), "eye_detector_model.pth")
print("Model saved as 'eye_detector_model.pth'")

