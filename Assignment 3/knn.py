import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_one(self, x):
        dists = np.sum((self.X - x)**2, axis=1)
        idx = np.argsort(dists)[:self.k]
        votes = self.y[idx]
        return 1 if np.sum(votes) >= (self.k / 2) else 0

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            preds.append(self.predict_one(X[i]))
        return np.array(preds)
