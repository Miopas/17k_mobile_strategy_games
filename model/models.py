from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
import numpy as np
import pdb

def get_prob(y_hat):
    ind = np.argsort(y_hat, axis=1)
    y_hat = np.take_along_axis(y_hat, ind, axis=1).tolist()
    y_hat = np.array([r[-1] for r in y_hat])
    return y_hat

def get_pred(y_prob):
    ret = []
    for i in y_prob.tolist():
        if (i[0] > i[1]):
            ret.append(0)
        else:
            ret.append(1)
    return np.array(ret)

class LogisticRegressionModel:
    def fit(self, X, y):
        self.clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)
        #print('\n'.join([str(x) for x in np.abs(self.clf.coef_[0]).tolist()]))

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X)), self.clf.predict_proba(X)

class SVMClassifier:
    def fit(self, X, y):
        self.clf = SVC(probability=True, gamma='auto').fit(X, y)

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X)), self.clf.predict_proba(X)


class LinearSVMClassifier:
    def fit(self, X, y):
        x_mu = X.mean(0)
        x_sigma = np.sqrt(X.var(0))
        X = (X - x_mu)/x_sigma
        self.clf = LinearSVC(random_state=0, tol=1e-5).fit(X, y)
        print('\n'.join([str(x) for x in np.abs(self.clf.coef_[0]).tolist()]))

    def predict(self, X):
        return self.clf.predict(X), None, None


class BoostingTree:
    def fit(self, X, y):
        self.clf = ensemble.GradientBoostingClassifier()
        self.clf.fit(X, y)
        #print('\n'.join([str(x) for x in self.clf.feature_importances_.tolist()]))

    def predict(self, X):
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X)), self.clf.predict_proba(X)


class MixModel:
    def fit(self, X, y):
        self.clf1 = ensemble.GradientBoostingClassifier().fit(X, y)
        self.clf2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

    def predict(self, X):
        alpha = 0.5
        y_prob = alpha * self.clf1.predict_proba(X) +  (1-alpha) * self.clf2.predict_proba(X)
        y_pred = get_pred(y_prob)
        y_prob = get_prob(y_prob)
        return y_pred, y_prob

class LinearModel:
    def fit(self, X, y):
        self.clf = ElasticNetCV(cv=5, random_state=0).fit(X, y)

    def predict(self, X):
        y_pred_prob = self.clf.predict(X)
        y_pred_prob_vec = np.array([[i, 1-i] for i in y_pred_prob])
        return _, _, y_pred_prob_vec


