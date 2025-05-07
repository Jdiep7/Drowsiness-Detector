import torch
import torch.nn as neural
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as py

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

data_dir='C:/Users/jdiep/projects/Drowsiness-Detector/train'
dataset = EyeDataSet(data_dir, transform)

image, label = dataset[100]

for image, label in dataset:
    break

dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataLoader:
    break

model = SimpleEyeClassifier(num_classes=2)
print(model)



