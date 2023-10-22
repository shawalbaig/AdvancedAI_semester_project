import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

color_folder = '/kaggle/input/plantvillage-dataset/color'
# grayscale_folder = '/kaggle/input/plantvillage-dataset/grayscale'
# segmented_folder = '/kaggle/input/plantvillage-dataset/segmented'

data = []
labels = []



def load_and_preprocess_images(folder, label):
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  
            data.append(img)
            labels.append(subfolder)
            
            
print("processing colors")
load_and_preprocess_images(color_folder, label='color')
#load_and_preprocess_images(grayscale_folder, label='grayscale')
#load_and_preprocess_images(segmented_folder, label='segmented')
print("done")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

print("Done")


def extract_color_features(images):
    color_features = []
    for image in images:
        # Calculate color histograms for each channel (R, G, B)
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate mean color values for each channel
        mean_color = np.mean(image, axis=(0, 1))
        
        # Flatten histograms
        hist_r = hist_r.flatten()
        hist_g = hist_g.flatten()
        hist_b = hist_b.flatten()
        
        # Concatenate features
        color_feature = np.concatenate((hist_r, hist_g, hist_b, mean_color))
        color_features.append(color_feature)
    return np.array(color_features)
print("extracting features from training and testing data")
# Extract color features for training and testing data
X_train_color_features = extract_color_features(X_train)
X_test_color_features = extract_color_features(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create the k-NN Classifier
k = 5 
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Training
print("Training the model")
knn_classifier.fit(X_train_color_features, y_train)

# Prediction
y_pred = knn_classifier.predict(X_test_color_features)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

