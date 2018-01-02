#Natural Language Processing

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
""" 3 for ignoring quotes"""

#Cleaning Text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    """it will remove all the punctuation and place space in their place"""
    review = review.lower()
    review = review.split()
    """ split the review into list of words """
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    """ it is used to remove irrelevent words like 'this' and also does stemming"""
    review = ' '.join(review)
    corpus.append(review)