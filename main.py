"""
Truly Native
__author__ : Rahul

"""

import pandas as pd;
import util;
from sklearn.feature_selection import VarianceThreshold;
import numpy as np;

#Read data
train = pd.read_csv("./data/train.csv.sample");

groups              = train.columns.to_series().groupby(train.dtypes).groups;
dataType_var        = {k.name: v for k, v in groups.items()};

datecolumns         = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', \
		       'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', \
                       'VAR_0179', 'VAR_0204', 'VAR_0217'];

categorical_f_card  = [];
features_uniq_count = [];
features_uniq_twice = [];

features_uniq_counts = [];

#Remove features which appear only once or twice
for i in train.columns:
  if i in datecolumns or i in ['ID', 'target']:
    continue;
  if i in dataType_var['int64'] or i in dataType_var['float64']:
     x       = train[i];
     no_unq = np.unique(x[~np.isnan(x)]).size; 
     if no_unq == 1:
       features_uniq_count.append(i);
     if no_unq == 2:
       features_uniq_twice.append(i);

     features_uniq_counts.append((i, no_unq));
  else:
     x       = train[i];
     no_uniq = np.unique(x.dropna()).size; 
     categorical_f_card.append((i,no_uniq));	

low_cardinal_categ_f  = [i[0] for i in categorical_f_card if i[1] < 10];
high_cardinal_categ_f = [i[0] for i in categorical_f_card if i[1] >= 10];

print(features_uniq_counts);

#Drop fields
train.drop('ID', axis=1, inplace=True);
train.drop(features_uniq_count, axis=1, inplace=True);
train.drop(datecolumns, axis=1, inplace=True); #Drop dates for now


#One hot encode
#train = util.encode_onehot(train, cols=categorical_features)






