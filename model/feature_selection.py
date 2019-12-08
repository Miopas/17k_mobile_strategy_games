from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
    while len(columns) > 0:
        col = columns[0]
        metric_max = -np.inf
        for col in columns:
            X_tmp = X_df.loc[:,col_sel + [col]]

            model = LogisticRegressionModel()
            model.fit(X_tmp, y)
            _, y_hat = model.predict(X_tmp)
            metric = roc_auc_score(y, y_hat)

            if metric > metric_max:
                metric_max = metric
                col_best = col

        columns.remove(col_best)
        col_sel.append(col_best)
        metric_lst.append(metric_max)
    return col_sel, metric_lst


if __name__ == '__main__':
    n = 100

    input_file = args.train_file
    df = pd.read_csv(input_file)
    columns = df.columns.tolist()
    columns.remove('Average User Rating')
    columns.remove('ID')

    #metric_max = -np.inf
    #col_sel_best = []
    #for i in range(n):
    #train, valid = train_test_split(df, test_size=0.2)
    #X_train, y_train = load_data(train)
    X_train, y_train = load_data(df)

    selected_cols, metric_list = stepwise(X_train, y_train, columns)

    metric_max = max(metric_list)
    max_idx = metric_list.index(metric_max)

    #X_valid, y_valid = load_data(valid)

    #_, y_hat = model.predict(X_valid)
    #metric = roc_auc_score(y_valid, y_hat)
    #if metric > metric_max:
    #    metric_max = metric
    #    col_sel_best = selected_cols

    print('best features:{}'.format(selected_cols[0:max_idx+1]))
    print('max auroc:{0:.3f}'.format(metric_max))
    #pdb.set_trace()

