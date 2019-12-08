import pandas as pd
import numpy  as np
import math

def load_data(df, col_sel=None):
    '''
    read features from input file
    input file format:

    return:  X numpy array, y numpy array
    '''
    X, y = [], []

    features = df.columns.tolist()
    features.remove('Average User Rating')
    features.remove('ID')
    label = 'Average User Rating'

    if col_sel != None:
        X = df.loc[:,col_sel]
        y = df[label].tolist()
    else:
        for index, row in df.iterrows():
            X.append([row[f] for f in features])
            y.append(row[label])
    return np.asarray(X), np.asarray(y)


