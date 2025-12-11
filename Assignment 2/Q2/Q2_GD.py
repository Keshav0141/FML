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
    step = 0.9 / L

    w = np.zeros(Xb.shape[1])
    norms = []
    for t in range(200):
        grad = (Xb.T @ (Xb @ w - y)) / n
        w -= step * grad
        norms.append(np.linalg.norm(w - w_ML))

    plt.plot(range(1, 201), norms)
    plt.xlabel("Iterations")
    plt.ylabel(r"||w_t - w_ML||_2")
    plt.title("Gradient Descent Convergence")
    plt.grid(True)
    plt.savefig("GD_norms.png", dpi=150)
    print("Gradient Descent done. Plot saved as GD_norms.png.")
