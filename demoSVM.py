import numpy as np
import csv
import statistics
from statistics import mean
from statistics import stdev
import pandas as pd
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from dbn.tensorflow import SupervisedDBNClassification


# Loading dataset
digits=pd.read_csv('pseudo.csv')
X=digits.iloc[1:,8:12].values
Y=digits.iloc[1:,13].values
#digits = load_digits()
#X, Y = digits.data, digits.target
print(Y)
# Data scaling
X = (X / 16).astype(np.float32)
#print(X.shape[0])
#print(X.shape[1])
#print(digits.iloc[1:,1].values)

#for i in range(1,X.shape[1]):
#    mean[i]=statistics.mean(digits.iloc[1:,i].values)
#for i in range(1,X.shape[1]):
#    sd[i]=statistics.stdev(digits.iloc[1:,i].values)
#for i in range(1,X.shape[1]):
#    for j in range(1,X.shape[0]):
#        X[j][i]= (X[j][i]-mean[i])/stdev[i]

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SVC()
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)

print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test,Y_pred))
