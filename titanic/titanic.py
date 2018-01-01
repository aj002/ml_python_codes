#titanic

#Implorting libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing training dataset
train = pd.read_csv('train.csv')
X = train.iloc[:, [2, 7]].values
y = train.iloc[:, 1].values

#Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting the training dataset to the classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the results
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
#Importing the test dataset
test = pd.read_csv('test.csv')
X_test = test.iloc[:, [1, 6]].values

#Predicting the results
y_pred = classifier.predict(X_test)

#Creating a CSV file
pd.DataFrame({'PassengerId': test.iloc[:, 0].values,
              'Survived' : y_pred}).set_index('PassengerId').to_csv('Submit_KNN_1.csv')
"""