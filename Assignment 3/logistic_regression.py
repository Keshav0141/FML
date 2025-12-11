# logistic_regression.py
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=200, batch_size=64, lambda_reg=0.001):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        # weighted binary cross entropy + L2 penalty
        loss = -np.sum(weights * (y_true * np.log(y_pred + eps) +
                                  (1 - y_true) * np.log(1 - y_pred + eps)))
        loss /= np.sum(weights)
        loss += self.lambda_reg * np.sum(self.weights ** 2)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # class weighting
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w_pos = n_samples / (2.0 * max(1, n_pos))
        w_neg = n_samples / (2.0 * max(1, n_neg))
        sample_weight = np.where(y == 1, w_pos, w_neg)

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                wb = sample_weight[batch_idx]

                linear = Xb @ self.weights + self.bias
                y_pred = self.sigmoid(linear)

                diff = (y_pred - yb) * wb
                db = np.sum(diff) / np.sum(wb)
                dw = (Xb.T @ diff) / np.sum(wb)

                dw += self.lambda_reg * self.weights

                dw = np.clip(dw, -5, 5)
                db = np.clip(db, -5, 5)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # loss for this epoch
            y_pred_full = self.sigmoid(X @ self.weights + self.bias)
            loss = self.compute_loss(y, y_pred_full, sample_weight)
            self.loss_history.append(loss)

    def predict(self, X):
        linear = X @ self.weights + self.bias
        probs = self.sigmoid(linear)
        return (probs >= 0.5).astype(int)
