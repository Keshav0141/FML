import numpy as np

def count_params_bernoulli(K, D):
    return K * D + (K - 1)

def count_params_gaussian(K, D):
    cov_params = D * (D + 1) / 2
    return int(K * (D + cov_params) + (K - 1))

def compute_aic_bic(log_likelihood, n_params, N):
    AIC = 2 * n_params - 2 * log_likelihood
    BIC = n_params * np.log(N) - 2 * log_likelihood
    return AIC, BIC
