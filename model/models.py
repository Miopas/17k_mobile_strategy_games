from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import ensemble
import numpy as np
import pdb

def get_prob(y_hat):
    ind = np.argsort(y_hat, axis=1)
    y_hat = np.take_along_axis(y_hat, ind, axis=1).tolist()
    y_hat = np.array([r[-1] for r in y_hat])
    return y_hat

class LogisticRegressionModel:
    def fit(self, X, y):
        self.clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


class SVMClassifier:
    def fit(self, X, y):
        #self.clf = SVC(kernel='linear', probability=True, gamma='auto').fit(X, y)
        self.clf = SVC(kernel='linear', probability=True, C=1.0).fit(X, y)
        #self.clf = LinearSVC(random_state=0, tol=1e-5).fit(X, y)

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


class BoostingTree:
    def fit(self, X, y):
        self.clf = ensemble.GradientBoostingClassifier()
        self.clf.fit(X, y)
        print('\n'.join([str(x) for x in self.clf.feature_importances_.tolist()]))

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


