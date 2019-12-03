from sklearn.linear_model import LogisticRegression

from data_loader import load_data

if __name__ == '__main__':
    X, y = load_data(return_X_y=True)
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)

    clf.predict(X[:2, :])
    clf.predict_proba(X[:2, :])
    clf.score(X, y)
