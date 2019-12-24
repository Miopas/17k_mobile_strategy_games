import sys
from data_loader import load_data
from sklearn import preprocessing
from models import LogisticRegressionModel, SVMClassifier, BoostingTree, MixModel
from evaluation import plot_confusion_matrix, plot_roc
import argparse
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn import metrics

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_file', type=str, help='training data')
parser.add_argument(
    '--test_file', type=str, help='testing data')

args = parser.parse_args()

models = {
    "lr": LogisticRegressionModel(),
    "svm": SVMClassifier(),
    "bt": BoostingTree(),
    "mix": MixModel(),
}

colors = {
    "lr": 'darkorange',
    "svm": 'deeppink',
    "bt": 'cornflowerblue',
}



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

if __name__ == '__main__':

    X_train, y_train = load_data(pd.read_csv(args.train_file))
    X_test, y_test = load_data(pd.read_csv(args.test_file))

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    import matplotlib.pyplot as plt
    modelnames = ['lr', 'svm', 'bt']
    
    y_test_vec = []
    for i in y_test.tolist():
        if i == 0:
            y_test_vec.append([1, 0])
        else:
            y_test_vec.append([0, 1])
    y_test_vec = np.array(y_test_vec)


    # Below for loop iterates through your models list
    for m in modelnames:
        model = models[m]
        color = colors[m]
        model.fit(X_train, y_train)
        y_pred, y_pred_prob, y_pred_prob_vec = model.predict(X_test)

    # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob_vec[:,1])
    # Calculate Area under the curve to display on the plot
        auc = metrics.roc_auc_score(y_test, y_pred_prob_vec[:,1])
    # Now, plot the computed values
        plt.plot(fpr, tpr, color=color, label='%s ROC (area = %0.2f)' % (m, auc))

    # Custom settings for the plot
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   # Display
    
