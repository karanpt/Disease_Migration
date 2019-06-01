import numpy as np
import csv
import statistics
from statistics import mean
from statistics import stdev
import pandas as pd
np.random.seed(1337)  
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from dbn.tensorflow import SupervisedDBNClassification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from tensorflow.python.keras import models

# Loading dataset
#digits = load_digits()
#X, Y = digits.data, digits.target
digits=pd.read_csv('classification_data.csv')
X=digits.iloc[1:,9:16].values.astype(float)
Y=digits.iloc[1:,17].values

# Data scaling
X = (X / 512).astype(np.float32)


# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#smt = SMOTE()
#X_train, Y_train = smt.fit_sample(X_train, Y_train)
#nr = NearMiss()
#X_train, Y_train = nr.fit_sample(X_train, Y_train)
# Training
#classifier = SupervisedDBNClassification(hidden_layers_structure=[8,8],
#                                         learning_rate_rbm=0.025,
#                                         learning_rate=0.1,
#                                         n_epochs_rbm=10,
#                                         n_iter_backprop=200,
#                                         batch_size=32,
#                                         activation_function='relu',
#                                         dropout_p=0.2)
#classifier.fit(X_train, Y_train)
#classifier.save('model.pkl')

# Restore it
classifier1 = SupervisedDBNClassification.load('model.pkl')
# Test
Y_pred = classifier1.predict(X_train)
print('Done.\nAccuracy: %f' % accuracy_score(Y_train, Y_pred))
print(confusion_matrix(Y_test,Y_pred))
