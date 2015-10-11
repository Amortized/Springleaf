import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import random;
 
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
 	  



