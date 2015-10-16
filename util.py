import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import random;
import time;
 
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def filterFeatures(df, cols, threshold):
    """
      cols is features which have only two unique values.
      For each feature, if any of those values is less than thres per, then ignore feature
    """
    remove_f = [];
    for i in cols:
       x = df[i].value_counts();  
       y = 100.0 * (x / sum(x.values));
       if (y < threshold).sum() == 1:
          #Ignore the feature
          remove_f.append(i);

    return remove_f;
 	  
def filterDateFeatures(df, cols, threshold):
    """
      For each date feature, if no of null's exceed a threshold.
    """
    remove_f = [];
    for i in cols:
       x = 100.0 * (df[i].isnull().sum() / float(df[i].size));
       if (x > threshold) :
          #Ignore the feature
          remove_f.append(i);

    return remove_f;

def createDateFeatures(df, cols):
    """
      #Convert each date column to year,month,season,date_of_month,day_of_week,weekend
    """
    new_features = [];
    for i in cols:	
       x = [time.strptime(str(k),"%d%b%y:%H:%M:%S") if str(k) != "nan" else "other" for k in df[i]];
       #For each date feature
       df[str(i) + "_year"]     = ["v_" + str(k.tm_year) if type(k) == time.struct_time else "v_" + str(-1) for k in x]
       df[str(i) + "_month"]    = ["v_" + str(k.tm_mon)  if type(k) == time.struct_time else "v_" + str(-1) for k in x]
       df[str(i) + "_weekday"]  = ["v_" + str(k.tm_wday) if type(k) == time.struct_time else "v_" + str(-1) for k in x]              
       df[str(i) + "_monthday"] = ["v_" + str(k.tm_mday) if type(k) == time.struct_time else "v_" + str(-1) for k in x]	 	
       df[str(i) + "_yearday"]  = [k.tm_yday if type(k) == time.struct_time else -1 for k in x]	 	

       new_features.append(str(i) + "_year");		
       new_features.append(str(i) + "_month");
       new_features.append(str(i) + "_weekday");
       new_features.append(str(i) + "_monthday");
       #new_features.append(str(i) + "_yearday");
   
    return df, new_features;


def binning(df, cols, threshold):
    """
      Features of interest
      For each feature, if any of those values is less than thres per, then all of them merged into one
    """

    for i in cols:
       x = df[i].value_counts();  
       y = 100.0 * (x / sum(x.values));
       z = y[y < threshold].keys();
       df[i] = ["other" if v in z else "v_" + str(v) for v in df[i].values];	
       
    return df;


    

