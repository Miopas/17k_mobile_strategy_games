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

    features = ['User Rating Count','Price','Languages','Size','Update_Gap','Name','Subtitle','age_rating','Puzzle','Simulation','Action','Board','In-app Purchases'];

    label = 'Average User Rating'
    for index, row in df.iterrows():
        if col_sel != None:
            X.append([row[f] for f in features if f in col_sel])
        else:
            X.append([row[f] for f in features])
        y.append(row[label])
    return np.asarray(X), np.asarray(y)


