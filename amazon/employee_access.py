#Implorting libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing training dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

trainx = train.iloc[ : , 1:10].values
trainy = train.iloc[ : , 0:1].values

testx = test.iloc[ : , 1:10].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainx = sc.fit_transform(trainx)
testx = sc.fit_transform(testx)


                                #ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(trainx, trainy, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
testy = classifier.predict(testx)
testy = 1*(testy > 0.5)
testy = testy.ravel()

#Creating a CSV file
pd.DataFrame({'Id': test['id'],
              'Action' : testy}).to_dense().to_csv('Submit_ANN_1.csv', index=False, 
                                                    columns=['Id','Action'])

