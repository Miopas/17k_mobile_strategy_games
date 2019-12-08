from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble
import numpy as np
import pdb

def get_prob(y_hat):
    ind = np.argsort(y_hat, axis=1)
    y_hat = np.take_along_axis(y_hat, ind, axis=1).tolist()
    y_hat = np.array([r[-1] for r in y_hat])
    return y_hat

#def get_class(y_hat):
#    y_hat = y_hat.tolist()
#    pdb.set_trace()
#    new = []
#    for y_lst in y_hat:
#        if y_lst[0] > y_lst[1]:
#            new.append(0)
#        else:
#            new.append(1)
#    return np.array(new)
#
#class LinearModel:
#    def fit(self, X, y):
#
#        n, m = X.shape
#
#        # normalize
#        x_mu = X.mean(0)
#        x_sigma = np.sqrt(X.var(0))
#        X = (X - x_mu)/x_sigma # normalized X
#
#        XTXinv = np.linalg.inv(np.dot(X.T, X))
#        self.beta = np.dot(XTXinv, np.dot(X.T, y))
#        self.intercept = np.mean(y)
#        return
#
#    def predict(self, X):
#        y_hat = np.dot(X, self.beta) + self.intercept
#        return get_class(y_hat), y_hat

class LogisticRegressionModel:
    def fit(self, X, y):
        self.clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)

    def predict(self, X):
        #return self.clf.predict(X)
        #return self.clf.predict(X), y_hat
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


class SVMClassifier:
    def fit(self, X, y):
        self.clf = SVC(probability=True, gamma='auto').fit(X, y)

    def predict(self, X):
        #return self.clf.predict(X)
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


class BoostingTree:
    def fit(self, X, y):
        self.clf = ensemble.GradientBoostingClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        #return self.clf.predict(X)
        return self.clf.predict(X), get_prob(self.clf.predict_proba(X))


