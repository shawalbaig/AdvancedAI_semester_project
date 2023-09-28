import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Define the paths to the training and validation directories
train_path = '/kaggle/input/plantvillage-dataset/color'
valid_path = '/kaggle/input/plantvillage-dataset/segmented'

# Get the list of classes (subdirectories) from the training directory
classes = os.listdir(train_path)

# Load the true labels for the validation set
y_test = [folder for folder in os.listdir(valid_path)]

# Determine the number of unique classes (labels)
num_classes = len(classes)

# Create a function for the random guessing model
def random_guessing_model(num_classes):
    predictions = [random.choice(classes) for _ in range(len(y_test))]
    return predictions

# Generate random predictions for the test set
random_predictions = random_guessing_model(num_classes)

# Calculate accuracy
accuracy = accuracy_score(y_test, random_predictions)
print(f"Accuracy: {accuracy}")

# Create a dictionary to store class label counts
class_counts = {cls: random_predictions.count(cls) for cls in classes}

# Create a dictionary to store the number of correct predictions for each class
correct_counts = {cls: 0 for cls in classes}

# Calculate the number of correct predictions
for true_label, pred_label in zip(y_test, random_predictions):
    if true_label == pred_label:
        correct_counts[true_label] += 1

# Create a pie chart to visualize the distribution of class labels
plt.figure(figsize=(8, 8))
plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Class Label Distribution")
plt.show()

# Create a bar graph to visualize the number of correct predictions for each class
plt.figure(figsize=(10, 6))
plt.bar(classes, [correct_counts[cls] for cls in classes], color='blue')
plt.xlabel('Class Label')
plt.ylabel('Number of Correct Predictions')
plt.title('Number of Correct Predictions for Each Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
