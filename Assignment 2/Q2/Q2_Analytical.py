import numpy as np

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

if __name__ == '__main__':
    train_file = "A2Q2Data_train - A2Q2Data_train.csv"
    X, y = load_data(train_file)
    Xb = add_bias(X)

    w_ML = np.linalg.pinv(Xb) @ y
    print("Analytical solution (w_ML) computed.")
    print(f"Shape: {w_ML.shape}")
