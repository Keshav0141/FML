import numpy as np

class PerceptronScratch:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_mod = np.where(y == 1, 1, -1)

        for _ in range(self.epochs):
            for i in range(n_samples):
                linear = np.dot(X[i], self.weights) + self.bias
                if y_mod[i] * linear <= 0:
                    self.weights += self.lr * y_mod[i] * X[i]
                    self.bias += self.lr * y_mod[i]

    def predict(self, X):
        linear = X @ self.weights + self.bias
        return (linear >= 0).astype(int)
