import sys
from data_loader import load_data
from sklearn import preprocessing
from models import LogisticRegressionModel, SVMClassifier, BoostingTree
from evaluation import plot_confusion_matrix
import argparse
import pandas as pd
import pdb

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    type=str,
    choices=['linear', 'lr', 'svm', 'bt', 'cnn', 'ft'],
    help='data file surfix')

parser.add_argument(
    '--train_file', type=str, help='training data')
parser.add_argument(
    '--valid_file', type=str, help='valid data')
parser.add_argument(
    '--test_file', type=str, help='testing data')

args = parser.parse_args()

models = {
    "lr": LogisticRegressionModel(),
    "svm": SVMClassifier(),
    "bt": BoostingTree(),
}


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

if __name__ == '__main__':

    #col_sel = ['User Rating Count', 'In-app Purchases', 'Puzzle', 'Subtitle', 'Board']
    #col_sel = ['User Rating Count', 'Subtitle']
    #X_train, y_train = load_data(pd.read_csv(args.train_file), col_sel)
    #X_test, y_test = load_data(pd.read_csv(args.test_file), col_sel)

    X_train, y_train = load_data(pd.read_csv(args.train_file))
    X_test, y_test = load_data(pd.read_csv(args.test_file))

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    model = models[args.model]
    model.fit(X_train, y_train)
    y_pred, y_pred_prob = model.predict(X_test)

    from sklearn.metrics import roc_auc_score
    print('auroc:{0:.3f}'.format(roc_auc_score(y_test, y_pred_prob)))

    from sklearn.metrics import accuracy_score
    print('acc:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print('precision_recall_fscore:{0:.3f}\t{1:.3f}\t{2:.3f}'.format(precision, recall, f1))

    _, cm = plot_confusion_matrix(y_test, y_pred, normalize=True, classes=le.classes_, figname=args.model+'.cm.png')

