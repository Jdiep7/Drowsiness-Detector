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
data_dir = 'C:/Users/jdiep/projects/Drowsiness-Detector/LocalizerDataset'

class BioIDEyeDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
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
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(scaled_eyes, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

dataset = BioIDEyeDataset(data_dir, transform)

train_size = int(0.8*len(dataset))
valid_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


       