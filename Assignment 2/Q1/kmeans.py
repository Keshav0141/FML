import numpy as np

def kmeans(X, K, max_iter=100, rng=None):
    if rng is None:
        rng = np.random
    N, D = X.shape
    centroids = X[rng.choice(N, K, replace=False)].astype(float)
    obj_history = []

    for it in range(max_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2) 
        labels = np.argmin(dists, axis=1)
        obj = np.sum(np.min(dists, axis=1))
        obj_history.append(obj)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            members = X[labels == k]
            if len(members) == 0:
                new_centroids[k] = X[rng.choice(N)]
            else:
                new_centroids[k] = members.mean(axis=0)
        centroids = new_centroids

    return np.array(obj_history), centroids, labels
