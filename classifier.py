import utils as utl
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.feature_selection import SelectKBest , f_classif
import matplotlib.pyplot as plt


tracks = utl.load ('C:/Users/Ahmad/Documents/tracks.csv' )
features = utl.load('C:/Users/Ahmad/Documents/features.csv' )

small = tracks['set', 'subset'] <= 'medium'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
X_train = features.loc[small & train, list(features.columns.levels[0]) ]
X_test = features.loc[small & test, list(features.columns.levels[0]) ]

print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

# Be sure training samples are shuffled.
X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

# Standardize features by removing the mean and scaling to unit variance.
scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)


selector = SelectKBest(f_classif, k=171)
X_new = selector.fit_transform(X_train, y_train)
X_tnew = selector.transform(X_test)
print (X_new.shape)
print (X_tnew.shape)

# Support vector classification.
clf = skl.svm.SVC()
clf.fit(X_new, y_train)
score = clf.score(X_tnew, y_test)
print('Accuracy: {:.2%}'.format(score))


