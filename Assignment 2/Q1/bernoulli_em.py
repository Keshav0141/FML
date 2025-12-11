import numpy as np
from logsumexp_utils import logsumexp

def bernoulli_em(X, K, max_iter=100, tol=1e-6, rng=None):
    
    if rng is None:
        rng = np.random
    N, D = X.shape
    
    pi = np.ones(K) / K
    mu = rng.uniform(0.25, 0.75, size=(K, D))
    ll_history = []

    for it in range(max_iter):
        # E-step (log domain)
        mu_clipped = np.clip(mu, 1e-12, 1 - 1e-12)
        log_mu = np.log(mu_clipped)        
        log_1m = np.log(1 - mu_clipped)     

        
        log_p = X @ log_mu.T + (1 - X) @ log_1m.T
        log_joint = log_p + np.log(pi + 1e-16)
        log_norm = logsumexp(log_joint, axis=1)     
        resp = np.exp(log_joint - log_norm[:, None])  

      
        ll = np.sum(log_norm)
        ll_history.append(ll)

        # M-step
        Nk = resp.sum(axis=0) + 1e-12   
        pi = Nk / N
        mu = (resp.T @ X) / Nk[:, None] 

       
        if it > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            ll_history += [ll_history[-1]] * (max_iter - len(ll_history))
            break

 
    if len(ll_history) < max_iter:
        ll_history += [ll_history[-1]] * (max_iter - len(ll_history))
    return np.array(ll_history), pi, mu
