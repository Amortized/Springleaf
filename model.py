"""

__author__ = 'amortized'
"""

import numpy  as np;
import random;
import sys;
import warnings;
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import copy
from sklearn.preprocessing import OneHotEncoder
from random import randint
from random import shuffle
import math
import copy
from sklearn.preprocessing import OneHotEncoder;
from numpy import array;
import pandas as pd;


#Read Data
train = pd.read_pickle("./data/train.csv.sample.pickle");

train_Y = train.target;
train.drop('target',inplace=True,axis=1);
train_X = train.as_matrix();

del train;

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.10, random_state=42);

dtrain      = xgb.DMatrix( X_train, label=Y_train, missing=float('NaN'));
dvalidation = xgb.DMatrix( X_test, label=Y_test,missing=float('NaN'));

print(X_train);

del X_train, X_test, Y_train, Y_test;

#Track metrics on the watchlist
watchlist = [ (dtrain,'train'), (dvalidation, 'validation') ];



#Params
param      = {'eval_metric' : 'auc', 'objective' : 'binary:logistic', 'nthread' : 8, \
	      'colsample_bytree' : 0.80, 'subsample' : 0.80,'max_depth' : 5, 'eta': 0.05};
num_round  = 1000;
classifier = xgb.train(param,dtrain,num_round,evals=watchlist,early_stopping_rounds=100);

metric     = classifier.best_score;
itr        = classifier.best_iteration;
print("\n Metric : " + str(metric) + " for Params " + str(param) + " occurs at " + str(itr));

classifier.save_model('./data/1.xgb_model');
f = open("./data/params.txt");
f.write("metric : " + str(metric) + "\n");
f.write("itr : " + str(itr) + "\n");
f.close();


