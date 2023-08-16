# Movie Genre Classification using Convolutional Neural Networks (CNN)

This code implements a CNN model using PyTorch to classify movie genres based on movie posters. The goal is to accurately predict the genre of a movie by analyzing its poster image.

## Data Preparation
The code loads the movie genre data and image titles from the "MovieGenre.csv" file. It then processes the genre information to create the target labels for training the CNN model. The data is split into a training set and a test set.

## Custom Dataset Class
A custom dataset class, `CustomImageDataset`, is defined to load the image data and their corresponding labels. This class utilizes the `torchvision.io.read_image` function to read the image files and applies image transformations such as resizing and normalization.

## CNN Model Architecture
The CNN model is defined in the `Net` class. It consists of several convolutional layers with leaky ReLU activation functions and max pooling layers for feature extraction. The output of the convolutional layers is flattened and fed into a fully connected layer for classification. The model architecture is as follows:
- Convolutional layer 1: 3 input channels, 15 output channels, kernel size 3x3
- Convolutional layer 2: 15 input channels, 30 output channels, kernel size 3x3
- Convolutional layer 3: 30 input channels, 30 output channels, kernel size 3x3
- Max pooling layer: kernel size 3x3
- Fully connected layer: 1470 input features, 9 output features (corresponding to the different movie genres)

## Training and Evaluation
The model is trained using the training dataset. The training loop iterates over the batches of data and performs forward and backward passes to calculate the loss and update the model's parameters using the Adam optimizer. After each epoch, the model is evaluated on the test dataset to measure its accuracy. The accuracy on the test set is printed for each epoch.

## Results
The model is trained for 30 epochs, and the accuracy on the test set is monitored to assess its performance. The training loss is also printed during the training process to track the model's convergence.

By training a CNN model on movie posters, this code aims to accurately predict the genre of a movie based solely on its visual representation.
