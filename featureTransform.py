"""
Truly Native
__author__ : Rahul

"""

import pandas as pd;
import util;
from sklearn.feature_selection import VarianceThreshold;
import numpy as np;

#Read data
train = pd.read_csv("/mnt/data/Springleaf/train.csv");

groups              = train.columns.to_series().groupby(train.dtypes).groups;
dataType_var        = {k.name: v for k, v in groups.items()};

datecolumns         = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', \
		       'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', \
                       'VAR_0179', 'VAR_0204', 'VAR_0217'];

categorical_f_card  = [];
features_uniq_count = [];
features_uniq_twice = [];

feature_freq_count  = [];


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
 
     feature_freq_count.append((i,no_unq));
  else:
     x       = train[i];
     no_unq = np.unique(x.dropna()).size; 
     categorical_f_card.append((i,no_unq));	

#Handle Date columns
date_columns_to_remove = util.filterDateFeatures(train, datecolumns, 95.0);
date_columns_to_encode = list(set(datecolumns) - set(date_columns_to_remove));

low_cardinal_categ_f  = [i[0] for i in categorical_f_card if i[1] < 10];
high_cardinal_categ_f = [i[0] for i in categorical_f_card if i[1] >= 10];

features_uniq_twice_removed = util.filterFeatures(train, features_uniq_twice, 1.0);
features_uniq_twice_keep    = list(set(features_uniq_twice) - set(features_uniq_twice_removed))

#Numerical features
numerical_f = list(set(train.columns) - set(datecolumns) - set(low_cardinal_categ_f) - \
              set(high_cardinal_categ_f) - set(['ID', 'target']) - set(features_uniq_twice) - \
	      set(features_uniq_count)); 	

#Numerical features Ineligible for binning
max_cardinality_bin = 8;
numerical_f_inelg_bin = [i[0] for i in feature_freq_count if i[1] > max_cardinality_bin];

print("No of features ineligible for binning : " + str(len(numerical_f_inelg_bin)));

numerical_f = list(set(numerical_f) - set(numerical_f_inelg_bin));

#Drop fields
train.drop('ID', axis=1, inplace=True);
train.drop(features_uniq_count, axis=1, inplace=True);
train.drop(features_uniq_twice_removed, axis=1, inplace=True);

train, date_columns_to_encode_ = util.createDateFeatures(train, date_columns_to_encode);
train.drop(datecolumns, axis=1, inplace=True); #Drop dates

train = util.binning(train, numerical_f, 0.5);
train = util.binning(train, high_cardinal_categ_f, 0.5);

#All features
all_f =  numerical_f + high_cardinal_categ_f + low_cardinal_categ_f + date_columns_to_encode_;

#One hot encode
train = util.encode_onehot(train, cols=all_f);

#Order of columns
train_cols = list(set(train.columns) - set('target'));

#Read test
test    = pd.read_csv("/mnt/data/Springleaf/test.csv");
id_df   = pd.DataFrame({'ID':test.ID});
id_df.to_csv("/mnt/data/Springleaf/test_ids.processed");

test.drop('ID', axis=1, inplace=True);
test.drop(features_uniq_count, axis=1, inplace=True);
test.drop(features_uniq_twice_removed, axis=1, inplace=True);

test, date_columns_to_encode_ = util.createDateFeatures(test, date_columns_to_encode);
test.drop(datecolumns, axis=1, inplace=True); #Drop dates for now

test = util.binning(test, numerical_f, 0.5);
test = util.binning(test, high_cardinal_categ_f, 0.5);

#All features
all_f =  numerical_f + high_cardinal_categ_f + low_cardinal_categ_f + date_columns_to_encode_;

#One hot encode
test     = util.encode_onehot(test, cols=all_f);

test_cols = list(set(test.columns));
 
#Take only features common between test and train
features = list(set(train_cols).intersection(set(test_cols))); 

print(list(features));

train      = train[features+['target']];
test       = test[features];

print("No of features after one hot encoding : " + str(len(train.columns)));
print("No of features after one hot encoding : " + str(len(test.columns)));

train.to_csv("/mnt/data/Springleaf/train.csv.processed");
test.to_csv("/mnt/data/Springleaf/test.csv.processed");

