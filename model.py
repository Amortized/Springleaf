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
import random;

#Read Data

train = pd.read_csv("/mnt/data/Springleaf/train.csv.processed");


train_Y = train.target;
train.drop('target',inplace=True,axis=1);
train_X = train.as_matrix();

del train;

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.10, random_state=42);

dtrain      = xgb.DMatrix( X_train, label=Y_train, missing=float('NaN'));
dvalidation = xgb.DMatrix( X_test, label=Y_test,missing=float('NaN'));


del X_train, X_test, Y_train, Y_test;

#Track metrics on the watchlist
watchlist = [ (dtrain,'train'), (dvalidation, 'validation') ];



#Params
param      = {'eval_metric' : 'auc', 'objective' : 'binary:logistic', 'nthread' : 16, \
	      'colsample_bytree' : 0.4, 'subsample' : 0.40,'max_depth' : 6, 'eta': 0.01,\
	      'seed' : random.randint(0,2000)};

num_round  = 3000;


classifier = xgb.train(param,dtrain,num_round,evals=watchlist,early_stopping_rounds=100);

metric     = classifier.best_score;
itr        = classifier.best_iteration;
print("\n Metric : " + str(metric) + " for Params " + str(param) + " occurs at " + str(itr));

del dtrain, dvalidation;


#Predict
t_x        = pd.read_csv("/mnt/data/Springleaf/test.csv.processed");
t_x        = t_x.as_matrix();
t_x        = xgb.DMatrix(t_x, missing=float('NaN'));
y_hat      = classifier.predict(t_x,ntree_limit=itr);


test_ids   = pd.read_csv("/mnt/data/Springleaf/test_ids.processed");
test_ids   = test_ids.ID;

df = pd.DataFrame({'ID':test_ids, 'target' : y_hat});
df.to_csv('submission2.csv', index=False);

