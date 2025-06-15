from itertools import combinations
from sklearn.svm import SVC


class DSVM:
    def __init__(self, class1, class2, **svm_params):
        self.class1 = class1
        self.class2 = class2
        self.model = SVC(**svm_params)

    def train(self, X, y):
        mask = (y == self.class1) | (y == self.class2)
        X_pair = X[mask]
        y_pair = y[mask]
        self.model.fit(X_pair, y_pair)

    def predict(self, X):
        return self.model.predict(X)
