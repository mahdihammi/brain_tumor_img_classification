This project focuses on the classification of brain tumors using Convolutional Neural Networks (CNN) and the Inception model and compare between them. The goal is to accurately distinguish between different types of brain tumors using MRI images.
# Dataset
The dataset used in this project consists of MRI images of brain tumors.
# Data Augmentation
Since the dataset is quite small, I did data augmentation to increase the size of the dataset, by introducing some changes and variations to the samples like zoom, shift, flip.
Here is an example of a result of data augmentation for a brain tumor images.
![image](https://github.com/mahdihammi/brain_tumor_img_classification/assets/89527502/d922b01f-8e63-482d-8c86-8d4310efedcf)
# Prepare Data
- Creating training and validations dirs using os path and mkdirs then split the data using a function takes source directory and training and validation directories and split size as parameters
- Making train and validation generators using ImageDataGenerator
# Modeling 
- Buit convolutional neural network (CNN) architecture using TensorFlow/Keras for image classification tasks. It consists of three convolutional layers followed by max pooling for feature extraction, then flattens the output for dense layers. Dropout is applied for regularization, and a sigmoid activation function is used in the output layer for binary classification. The model is compiled with Adam optimizer, binary crossentropy loss, and accuracy metrics.
- pre-trained InceptionV3 model for transfer learning in TensorFlow/Keras. It loads weights from a specified file (local_weights_file) and freezes all layers to prevent further training. The model is configured to exclude the top classification layers (include_top=False) and expects input images of size 150x150 pixels with 3 channels (RGB).
