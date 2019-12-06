import pandas as pd
import numpy  as np
import math

def load_data(input_file):
    '''
    read features from input file
    input file format:

    return:  X numpy array, y numpy array
    '''
    X, y = [], []

    features = ['User Rating Count','Price','Languages','Size','Update_Gap','Name','Subtitle','age_rating','Puzzle','Simulation','Action','Board','In-app Purchases'];

    label = 'Average User Rating'
    df = pd.read_csv(input_file)
    for index, row in df.iterrows():
        X.append([row[f] for f in features])
        #y.append(math.ceil(row[label]))
        y.append(row[label])
    return np.asarray(X), np.asarray(y)


