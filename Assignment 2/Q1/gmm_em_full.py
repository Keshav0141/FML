import numpy as np
from logsumexp_utils import logsumexp

def gaussian_em_full(X, K, max_iter=100, tol=1e-6, rng=None, reg_covar=1e-6):
    if rng is None:
        rng = np.random
    N, D = X.shape
    pi = np.ones(K) / K
    means = X[rng.choice(N, K, replace=False)].astype(float)
    emp_cov = np.cov(X, rowvar=False) + reg_covar * np.eye(D)
    covs = np.array([emp_cov.copy() for _ in range(K)])
    ll_history = []

    for it in range(max_iter):
        log_prob = np.zeros((N, K))
        for k in range(K):
            mu = means[k]
            cov = covs[k] + reg_covar * np.eye(D)
            # ensure PD
            sign, logdet = np.linalg.slogdet(cov)
            invcov = np.linalg.inv(cov)
            dif = X - mu
            quad = np.sum((dif @ invcov) * dif, axis=1)
            log_prob[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + quad)

        log_joint = log_prob + np.log(pi + 1e-16)
        log_norm = logsumexp(log_joint, axis=1)
        resp = np.exp(log_joint - log_norm[:, None])
        ll = np.sum(log_norm)
        ll_history.append(ll)

        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / N
        means = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            dif = X - means[k]
            covs[k] = (resp[:, k][:, None] * dif).T @ dif / Nk[k] + reg_covar * np.eye(D)

        if it > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            ll_history += [ll_history[-1]] * (max_iter - len(ll_history))
            break

    if len(ll_history) < max_iter:
        ll_history += [ll_history[-1]] * (max_iter - len(ll_history))
    return np.array(ll_history), pi, means, covs
