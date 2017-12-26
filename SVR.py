#SVR

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(np.reshape(y, (-1, 1))))

#Fitting the dataset to SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Predicting a new result with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing SVR
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Salary vs Position By Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
