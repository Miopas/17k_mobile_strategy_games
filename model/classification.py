import sys
from data_loader import load_data
from sklearn import preprocessing
from models import LogisticRegressionModel, SVMClassifier, BoostingTree
from evaluation import plot_confusion_matrix
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    type=str,
    choices=['lr', 'svm', 'bt', 'cnn', 'ft'],
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

if __name__ == '__main__':
    X_train, y_train = load_data(args.train_file)
    X_test, y_test = load_data(args.test_file)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    model = models[args.model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _, cm = plot_confusion_matrix(y_test, y_pred, normalize=True, classes=le.classes_, figname=args.model+'.cm.png')

    #from sklearn.metrics import precision_recall_fscore_support
    #print(precision_recall_fscore_support(y_test, y_pred))

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))


