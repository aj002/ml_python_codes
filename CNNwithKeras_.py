#CNN with Keras

#Load the dataset
from keras.datasets import fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

#Analyse the dataset
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('Training data shape: ', train_X.shape, train_Y.shape)
print('Test data shape: ', test_X.shape, test_Y.shape)

#Finding the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of output: ', nClasses)
print('Output classes: ', classes)

plt.figure(figsize = [5, 5])

#Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0, :, :], cmap = 'gray')
plt.title('Ground Truth : {}'.format(train_Y[0]))

#Display the first image in test data
plt.subplot(122)
plt.imshow(test_X[0, :, :], cmap = 'gray')
plt.title('Ground Truth : {}'.format(test_Y[0]))

#Data preprocessing
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_Y.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255
test_X = test_X/255

#Change the labels from categorical to one hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

#Display the change for category label using one-hot encoding
print('Original label: ', train_Y[0])
print('After conversion to one-hot: ', train_Y_one_hot[0])

#Splitting the data set
from sklearn.model_selection import train_test_split
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size =0.2, random_state = 13)

#Model the data
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

#Neural Network Architecture