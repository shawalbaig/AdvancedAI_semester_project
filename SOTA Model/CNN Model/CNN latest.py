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

# dataset
dataset_path = '/kaggle/input/plantvillage-dataset/color'

# Gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc1 = nn.Linear(32, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1]
])

# Load the dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and validation sets (80/20 split)
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

# Initialize the model and move it to the device (GPU or CPU)
num_classes = len(full_dataset.classes)
model = CNNModel(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss and accuracy values for plotting
losses = []
accuracies = []

print("Beginning Training!")

# Training
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Compute accuracy for the entire epoch
    epoch_accuracy = 100.0 * correct_predictions / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy: {epoch_accuracy:.2f}%")

    # Save loss and accuracy values for plotting
    losses.append(running_loss / len(train_loader))
    accuracies.append(epoch_accuracy)

print("Training completed!")


# Save the final model state to a file
save_path = "model.pth"
torch.save(model.state_dict(), save_path)
print("Model saved to:", save_path)


# Plot loss vs accuracy chart
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epoch')

plt.tight_layout()
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Set the model to evaluation mode
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through data loader to get predictions
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(predicted.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Weight Histograms
# Function to plot weight histograms for each layer
def plot_weight_histograms(model):
    plt.figure(figsize=(12, 6))
    plt.suptitle('Weight Histograms')

    for idx, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            plt.subplot(2, 4, idx + 1)
            plt.hist(param.data.cpu().numpy().flatten(), bins=30, color='blue', alpha=0.7)
            plt.title(name)

    plt.tight_layout()
    plt.show()

# Plot weight histograms
plot_weight_histograms(model)

# Feature Map Visualization
# Function to visualize feature maps of a specific layer
def visualize_feature_maps(model, layer_num, image):
    # Set the model to evaluation mode
    model.eval()

    # Get the specified layer
    target_layer = model.conv1  # Change this to the desired layer

    # Forward pass to get feature maps
    activations = target_layer(image.unsqueeze(0).to(device))
    
    # Plot the feature maps
    plt.figure(figsize=(12, 6))
    for i in range(activations.size(1)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(activations[0, i].cpu().detach().numpy(), cmap='viridis')
        plt.title(f'Feature Map {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Choose an image from the dataset
image, label = train_dataset[0]

# Visualize feature maps for the first convolutional layer
visualize_feature_maps(model, 1, image.to(device))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
model.eval()  # Set the model to evaluation mode

true_labels = []
predicted_labels = []

with torch.no_grad():  # Disable gradient tracking for evaluation
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
class_names = [class_name for class_name in full_dataset.classes]
report = classification_report(true_labels, predicted_labels, target_names=class_names)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)
