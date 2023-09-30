import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model

print("Beginning...")

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# re-size all the images to this
IMAGE_SIZE = [256, 256]

# Path to the color dataset
data_dir = '/kaggle/input/plantvillage-dataset/color'

# Get the list of class folders (plant disease categories)
class_folders = os.listdir(data_dir)

# Create empty lists for image paths and labels
image_paths = []
labels = []

# Iterate through each class folder and collect image paths and labels
print("Getting and Splitting Data...")
for class_folder in class_folders:
    class_path = os.path.join(data_dir, class_folder)
    image_files = os.listdir(class_path)
    
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        image_paths.append(image_path)
        labels.append(class_folder)

# Split the data into training and validation sets (80% train, 20% validation)
train_paths, valid_paths, train_labels, valid_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Create directories for training and validation data
train_path = '/kaggle/working/train'
valid_path = '/kaggle/working/valid'


#os.makedirs(train_path, exist_ok=True)
#os.makedirs(valid_path, exist_ok=True)

# Copy training images to the 'train' directory
for image_path, label in zip(train_paths, train_labels):
    dest_dir = os.path.join(train_path, label)
    os.makedirs(dest_dir, exist_ok=True)
    os.symlink(image_path, os.path.join(dest_dir, os.path.basename(image_path)))

# Copy validation images to the 'valid' directory
for image_path, label in zip(valid_paths, valid_labels):
    dest_dir = os.path.join(valid_path, label)
    os.makedirs(dest_dir, exist_ok=True)
    os.symlink(image_path, os.path.join(dest_dir, os.path.basename(image_path)))

print("Data Loaded, Loading Model...")

# Load the pre-trained ResNet50 model
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = True

x = Flatten()(resnet.output)

# Define the model architecture
prediction = Dense(38, activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1
)

# Load and preprocess the training data
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load and preprocess the test data
test_set = train_datagen.flow_from_directory(
    valid_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
print("Training Model...")
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Plot training loss and validation loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss')  # Save the figure
plt.show()

# Plot training accuracy and validation accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc')  # Save the figure
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the layer name for feature map extraction
layer_name = 'conv1_conv'

# Create a model to extract feature maps from the chosen layer
feature_map_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Replace 'image_path' with the path to your dataset image
image_path = '/kaggle/input/plantvillage-dataset/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG'
input_image = load_img(image_path, target_size=(256, 256))
input_image = img_to_array(input_image)
input_image = np.expand_dims(input_image, axis=0)

# Generate feature maps for the input image
feature_maps = feature_map_model.predict(input_image)

# Print the shape of the feature maps (for debugging)
print("Feature Maps Shape:", feature_maps.shape)

# Create a 4x4 grid for displaying the final 16 feature maps
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')  # Display a single feature map
    plt.title(f"Feature Map {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

# Get the true labels
true_labels = test_set.classes

# Get the predicted labels
predicted_labels = model.predict(test_set)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=test_set.class_indices.keys(), output_dict=True)

# Extract precision and recall values from the report
precision_values = [report[label]['precision'] for label in test_set.class_indices.keys()]
recall_values = [report[label]['recall'] for label in test_set.class_indices.keys()]

# Plot precision and recall as bar charts
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(test_set.class_indices.keys(), precision_values)
plt.xlabel('Classes')
plt.ylabel('Precision')
plt.title('Precision per Class')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
plt.bar(test_set.class_indices.keys(), recall_values)
plt.xlabel('Classes')
plt.ylabel('Recall')
plt.title('Recall per Class')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# Save the trained model
model.save('/kaggle/working/resnet_plant_disease_model.h5')

# Save the model's weights only
model.save_weights('/kaggle/working/resnet_plant_disease_model_weights.h5')
