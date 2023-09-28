import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# re-size all the images to this
IMAGE_SIZE = [256, 256]

train_path = '/kaggle/input/plantvillage-dataset/color'
valid_path = '/kaggle/input/plantvillage-dataset/segmented'
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


for layer in resnet.layers:
    layer.trainable = True

x = Flatten()(resnet.output)

data_dir = '/kaggle/input/plantvillage-dataset/grayscale'

class_folders = os.listdir(data_dir)
image_paths = []
labels = []

for class_folder in class_folders:
    class_path = os.path.join(data_dir, class_folder)
    image_files = os.listdir(class_path)
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        image_paths.append(image_path)
        labels.append(class_folder)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

prediction = Dense(38, activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, 
                             zoom_range=0.2, horizontal_flip=True, 
                             fill_mode="nearest", validation_split=0.1)


def train_test_split(dataset, test_size=0.2):
    length = len(dataset)
    train_length = round(length * (1 - test_size))
    test_length = length - train_length
    
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])
    return train_dataset, test_dataset



training_set = train_datagen.flow_from_directory('/kaggle/input/plantvillage-dataset/color',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 subset='training'
                                                )

test_set = train_datagen.flow_from_directory('/kaggle/input/plantvillage-dataset/color',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            subset='validation',
                                           )
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt

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
layer_name = 'conv5_block3_out'

# Create a model to extract feature maps from the chosen layer
feature_map_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Replace 'image_path' with the path to your dataset image
image_path = '/kaggle/input/plantvillage-dataset/grayscale/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG'
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
