from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import argparse
import pdb

from models import LogisticRegressionModel, SVMClassifier, BoostingTree
from data_loader import load_data

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_file', type=str, help='training data')
args = parser.parse_args()


def stepwise(X, y, headers):
    n, m = X.shape

    columns = [col for col in headers]
    X_df = pd.DataFrame(X, columns=columns)

    col_sel = []
    metric_lst = []
    model_lst = []
    while len(columns) > 0:
        col = columns[0]
        metric_max = -np.inf
        model_max = None
        for col in columns:
            X_tmp = X_df.loc[:,col_sel + [col]]

            #model = LogisticRegressionModel()
            model = BoostingTree()
            model.fit(X_tmp, y)
            y_pred, y_pred_prob = model.predict(X_tmp)
            metric = accuracy_score(y, y_pred)

            if metric > metric_max:
                metric_max = metric
                model_max = model
                col_best = col

        columns.remove(col_best)
        col_sel.append(col_best)
        metric_lst.append(metric_max)
        model_lst.append(model_max)
    return col_sel, metric_lst, model_lst


if __name__ == '__main__':
    n = 20

    input_file = args.train_file
    df = pd.read_csv(input_file)
    columns = df.columns.tolist()
    columns.remove('Average User Rating')
    columns.remove('ID')

    metric_max = -np.inf
    col_sel_best = []
    for i in range(n):
        train, valid = train_test_split(df, test_size=0.2)
        X_train, y_train = load_data(train)

        selected_cols, metric_list, model_list = stepwise(X_train, y_train, columns)

        metric = max(metric_list)
        idx = metric_list.index(metric)
        col_sel = selected_cols[0:idx+1]

        X_valid, y_valid = load_data(valid, col_sel)

        y_pred, y_pred_prob = model_list[idx].predict(X_valid)
        metric = accuracy_score(y_valid, y_pred)
        if metric > metric_max:
            metric_max = metric
            col_sel_best = col_sel


    print('best features:{}'.format(col_sel_best))
    print('max acc:{0:.3f}'.format(metric_max))

