import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

dataset_path = '/kaggle/input/plantvillage-dataset/color'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 0.001
num_epochs = 50

class CNNModel(nn.Module):

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

num_classes = len(full_dataset.classes)
model = CNNModel(num_classes).to(device)
