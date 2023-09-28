# Plant Disease Detection: A Comparative Analysis of Machine Learning Models

## Abstract
This research paper outlines a comparative analysis of various image classification models in the domain of agriculture with regard to plant disease detection. This study encompasses 4 distinct image classification models, namely, a Naive Bayes Model (naive model), a K-Nearest Neighbours model (baseline model), a Convolutional Neural Network (SOTA model), and a Residual Neural Network (SOTA model). These models are evaluated and compared using two separate datasets, namely the "PlantVillage Dataset" and the "New Plant Disease Dataset". The outcomes of this research paper are anticipated to provide deeper insight with regard to the suitability and performance of these image classification models for plant disease detection.

## Introduction
In our approach to tackling food security challenges in staple cereals, we combine foundational and advanced machine learning models. To ensure effective disease detection, we begin with two foundational models:

- **Naive Bayes Model**: The Naive Bayes model simplifies assumptions about feature independence and utilizes probabilistic calculations based on the data to make predictions, rather than considering complex feature interactions.
- **K-Nearest Neighbours Model**: The baseline model employs the k-nearest neighbour algorithm, which classifies samples based on the majority class among its k-nearest neighbours in the feature space.

These foundational models are the starting point in our comprehensive strategy for enhancing crop disease detection. Furthermore, we are going to employ advanced models:

- **Convolutional Neural Network (CNN)**: A type of deep learning architecture designed for computer vision applications that automatically learn hierarchical representations from raw data.
- **Residual Network (ResNet) Model**: A type of CNN architecture that introduced the concept of residual learning through skip connections to address the vanishing gradient problem, enabling deeper learning and improved accuracy for image classification tasks.

In our endeavor to validate and falsify the results of our diverse machine learning models, we will utilize a variety of robust evaluation methods and visualization techniques. For the foundational models, namely the Naive Bayes model and the k-nearest neighbors (k-NN) model, we will use their accuracy, recall, precision, and F1-Score tables, alongside their overall accuracy in order to gain a deeper understanding of these model's efficiency, performance, and robustness.

When evaluating the Convolutional Neural Network (CNN) model, we will plot learning curves to monitor its training and validation accuracy/loss throughout their epochs, allowing us to detect signs of improvement alongside any issues of overfitting and underfitting. We will also use Weight histograms to quantify and Feature Maps to visualize the regions that the models deem most relevant for their predictions, providing a deeper understanding of its feature detection capabilities.

Additionally, to comprehensively evaluate the performance of the Residual Network (ResNet) model, we employed various metrics and visualizations. We started by visualizing the feature maps, providing insights into the model's understanding of the image data. Additionally, we examined the training and validation accuracy curves as well as the training and validation loss curves, allowing us to assess the model's learning progress and generalization. Furthermore, we analyzed recall and precision metrics for each class, providing a detailed understanding of the model's performance on individual disease categories. These metrics and visualizations collectively offer a thorough assessment of the ResNet architecture's decision-making process and its effectiveness in disease classification.

Moreover, we aim to evaluate the performance of our models on real-world data collected through a mobile application. By allowing users to upload plant pictures taken in practical scenarios, we can assess how well the models generalize to these real-world conditions and make predictions on previously unseen images. This validation process will provide valuable insights into the models' effectiveness and robustness in practical use cases.