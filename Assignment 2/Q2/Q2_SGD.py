import numpy as np
import matplotlib.pyplot as plt

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

if __name__ == "__main__":
    train_file = "A2Q2Data_train - A2Q2Data_train.csv"
    X, y = load_data(train_file)
    Xb = add_bias(X)
    w_ML = np.linalg.pinv(Xb) @ y

    n = Xb.shape[0]
    sigma_max = np.linalg.svd(Xb, compute_uv=False)[0]
    L = (sigma_max ** 2) / n
    step_sgd = 0.1 * (0.9 / L)
    batch_size = 100
    epochs = 50

    w = np.zeros(Xb.shape[1])
    norms = []
    rng = np.random.default_rng(42)

    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            xb = Xb[perm[i:i+batch_size]]
            yb = y[perm[i:i+batch_size]]
            grad = (xb.T @ (xb @ w - yb)) / batch_size
            w -= step_sgd * grad
            norms.append(np.linalg.norm(w - w_ML))

    plt.plot(range(1, len(norms) + 1), norms)
    plt.xlabel("SGD updates")
    plt.ylabel(r"||w_t - w_ML||_2")
    plt.title("SGD (batch=100) Convergence")
    plt.grid(True)
    plt.savefig("SGD_norms.png", dpi=150)
    print("SGD done. Plot saved as SGD_norms.png.")
