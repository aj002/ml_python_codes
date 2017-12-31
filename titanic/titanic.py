#titanic

#Implorting libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing training dataset
train = pd.read_csv('train.csv')
X_train = train.iloc[:, [2, 7]].values
y_train = train.iloc[:, 1].values

#Fitting the training dataset to the classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, y_train)

#Importing the test dataset
test = pd.read_csv('test.csv')
X_test = test.iloc[:, [1, 6]].values

#Predicting the results
y_pred = classifier.predict(X_test)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#Creating a CSV file
pd.DataFrame({'PassengerId': test.iloc[:, 0].values,
              'Survived' : y_pred}).set_index('PassengerId').to_csv('Submit_KNN_1.csv')