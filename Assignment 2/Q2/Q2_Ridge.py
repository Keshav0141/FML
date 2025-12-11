import numpy as np
import matplotlib.pyplot as plt

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def ridge_closed_form(Xb, y, lam):
    n, d = Xb.shape
    return np.linalg.solve(Xb.T @ Xb + n * lam * np.eye(d), Xb.T @ y)

def k_fold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.array_split(perm, k)

if __name__ == "__main__":
    train_file = "A2Q2Data_train - A2Q2Data_train.csv"
    test_file = "A2Q2Data_test - A2Q2Data_test.csv"
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)
    lambdas = np.logspace(-6, 2, 30)
    folds = k_fold_indices(X_train_b.shape[0], 5)

    val_mses = []
    for lam in lambdas:
        fold_mses = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(5) if j != i])
            X_tr, y_tr = X_train_b[train_idx], y_train[train_idx]
            X_va, y_va = X_train_b[val_idx], y_train[val_idx]
            w = ridge_closed_form(X_tr, y_tr, lam)
            fold_mses.append(np.mean((y_va - X_va @ w) ** 2))
        val_mses.append(np.mean(fold_mses))

    best_idx = np.argmin(val_mses)
    best_lambda = lambdas[best_idx]
    w_R = ridge_closed_form(X_train_b, y_train, best_lambda)
    w_ML = np.linalg.pinv(X_train_b) @ y_train

    test_mse_ml = np.mean((y_test - X_test_b @ w_ML) ** 2)
    test_mse_r = np.mean((y_test - X_test_b @ w_R) ** 2)

    plt.semilogx(lambdas, val_mses, marker="o")
    plt.xlabel("lambda (log scale)")
    plt.ylabel("Validation MSE")
    plt.title("Ridge Regression Cross-Validation")
    plt.grid(True)
    plt.savefig("ridge_cv.png", dpi=150)

    print(f"Best lambda: {best_lambda:.6g}")
    print(f"Test MSE (w_ML): {test_mse_ml:.6f}")
    print(f"Test MSE (w_R): {test_mse_r:.6f}")
    print("Plot saved as ridge_cv.png.")
