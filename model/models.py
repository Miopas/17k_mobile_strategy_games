from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble
import numpy as np


class LinearModel:
    def predict(self, X, y):
        XTXinv = np.linalg.inv(np.dot(X.T, X))
        beta = np.dot(XTXinv, np.dot(X.T, y))
        y_hat = np.dot(X, beta)
        mse = np.mean(np.square(y - y_hat))
        return mse


class LogisticRegressionModel:
    def fit(self, X, y):
        self.clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class SVMClassifier:
    def fit(self, X, y):
        self.clf = SVC(gamma='auto').fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class BoostingTree:
    def fit(self, X, y):
        self.clf = ensemble.GradientBoostingClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


