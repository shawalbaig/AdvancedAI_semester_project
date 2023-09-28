import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Define the path to dataset
dataset_path = '/kaggle/input/plantvillage-dataset/color'

# Initialize lists
data = []
labels = []

print(f"Loading Images...")
# Load images and labels from the dataset
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    # Iterate through images in each class folder
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        
        # Read and resize the image (you can perform additional preprocessing here)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))  # Adjust the size as needed
        
        # Flatten the image into a 1D array (feature vector)
        image = image.flatten()
        
        # Append the feature vector and label to the data lists
        data.append(image)
        labels.append(class_name)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the dataset into training and testing sets
print(f"Splitting Images...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Beginning Training...")
# Initialize and train the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


#Charts
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Create a bar chart of class distribution
class_counts = {class_name: list(y_test).count(class_name) for class_name in np.unique(y_test)}
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=90)
plt.show()


#Feature Importance Plot
import numpy as np
import matplotlib.pyplot as plt

# Calculate the standard deviations of features for each class
class_std_devs = {}
for class_name in np.unique(y_train):
    class_indices = np.where(y_train == class_name)
    class_data = X_train[class_indices]

    std_devs = np.std(class_data, axis=0)

    class_std_devs[class_name] = std_devs

# Calculate the mean standard deviation across all classes for each feature
mean_std_devs = np.mean(list(class_std_devs.values()), axis=0)

# Sort features by mean standard deviation in descending order
sorted_indices = np.argsort(mean_std_devs)[::-1]
sorted_features = X_train[:, sorted_indices]

# Plot the feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(X_train.shape[1]), mean_std_devs[sorted_indices])
plt.xlabel('Feature Index')
plt.ylabel('Mean Standard Deviation')
plt.title('Feature Importance Plot')
plt.xticks(range(X_train.shape[1]), sorted_indices)
plt.show()