# Foundations of Machine Learning – Assignment 2  
**Author:** Aryan Prasad  
**Roll No.:** DA25M007  

This repository contains code and analysis for two major tasks:

1. Mixture Models and Expectation–Maximization (EM)  
2. Linear Regression using Analytical Solution, GD, SGD, and Ridge Regression  

---

# Question 1: Mixture Models and EM Algorithm

## Dataset Description  
The file `A2Q1.csv` contains 400 samples in $\{0,1\}^{50}$, representing flattened $10 \times 5$ binary images.  
Binary data is best modeled using a **Mixture of Bernoulli Distributions**.

---

# (i) Bernoulli Mixture Model and EM Derivation

## Generative Model  

Cluster assignment:

$$
z_i \sim \text{Categorical}(\pi_1,\dots,\pi_K)
$$

Given $z_i = k$:

$$
x_{id} \sim \text{Bernoulli}(\theta_{kd})
$$

### Likelihood under component $k$

$$
p(x_i \mid z_i = k)
=
\prod_{d=1}^{D}
\theta_{kd}^{\,x_{id}}
(1-\theta_{kd})^{\,1-x_{id}}
$$

### Mixture likelihood

$$
p(x_i)
=
\sum_{k=1}^{K}
\pi_k
\prod_{d=1}^{D}
\theta_{kd}^{\,x_{id}}
(1-\theta_{kd})^{\,1-x_{id}}
$$

---

# E-Step  

Responsibilities:

$$
r_{ik} =
\frac{
\pi_k
\prod_{d}
\theta_{kd}^{\,x_{id}}
(1-\theta_{kd})^{\,1-x_{id}}
}{
\sum_{j=1}^{K}
\pi_j
\prod_{d}
\theta_{jd}^{\,x_{id}}
(1-\theta_{jd})^{\,1-x_{id}}
}
$$

---

# M-Step  

Mixing weights:

$$
\pi_k = \frac{N_k}{N}, \quad 
N_k = \sum_{i=1}^{N} r_{ik}
$$

Bernoulli means:

$$
\theta_{kd} = \frac{\sum_{i=1}^{N} r_{ik} x_{id}}{N_k}
$$

---

## Bernoulli EM Results  
- Log-likelihood rises rapidly and stabilizes near **6600**.  
- Component probability maps reveal meaningful binary patterns.

---

# (ii) Gaussian Mixture Model (GMM)

Gaussian likelihood:

$$
p(x_i \mid z_i=k)
=
\mathcal{N}(x_i \mid \mu_k, \Sigma_k)
$$

### Observation  
- Achieves higher numerical likelihood  
- Completely inappropriate for binary data  
- BIC severely penalizes covariance parameters → overfitting

---

# (iii) K-Means Clustering

Objective minimized:

$$
J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \|x_i - \mu_k\|^2
$$

### Conclusion  
- Fast  
- Hard clusters only  
- Not a probabilistic model  

---

# (iv) Model Comparison using AIC & BIC  

Bernoulli mixture:

- Log-likelihood: **−6622.13**  
- AIC: **13650.25**  
- BIC: **14460.52**

Gaussian mixture:

- Log-likelihood: **+3629.95**  
- Very large BIC (due to covariance parameters)

### Final Choice  
**Bernoulli EM** is the best model for this dataset.

---

# Question 2: Regression — Analytical, GD, SGD, Ridge

We use the squared loss:

$$
L(w) = \frac{1}{2N} \|X w - y\|_2^2
$$

A bias term is added to obtain $101$-dimensional weight vectors.

---

# (i) Analytical Least-Squares Solution

Closed-form solution:

$$
w_{\text{ML}} = (X^T X)^{-1} X^T y
$$

If singular:

$$
w_{\text{ML}} = X^{+} y
$$

---

# (ii) Gradient Descent (GD)

Gradient:

$$
\nabla L(w) = \frac{1}{N} X^T(Xw - y)
$$

Update rule:

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

Stability condition:

$$
0 < \eta < \frac{2}{\lambda_{\max}}
$$

### GD Convergence  

The curve  
$$
\|w_t - w_{\text{ML}}\|_2
$$  
decreases smoothly from **1.55 → 0.88** over 200 iterations.

---

# (iii) Stochastic Gradient Descent (SGD)

Mini-batch size: $100$  

SGD update:

$$
w_{t+1} =
w_t - \eta_{\text{SGD}}
\frac{1}{b}
X_B^T(X_B w - y_B)
$$

### SGD Convergence  

The quantity  
$$
\|w_t - w_{\text{ML}}\|_2
$$  
drops from **1.6 → 0.39** over ~5000 updates.

---

# (iv) Ridge Regression

Objective:

$$
L_\lambda(w) =
\frac{1}{2N} \|X w - y\|_2^2
+
\frac{\lambda}{2} \|w\|_2^2
$$

Closed-form:

$$
w_R = (X^T X + N \lambda I)^{-1} X^T y
$$

### Cross-Validation  

$\lambda$ values range from $10^{-6}$ to $10^2$.

Best choice:

$$
\hat{\lambda} \approx 1.6 \times 10^{-4}
$$

### Test Errors  

- $\text{MSE}(w_{\text{ML}}) = 0.370752$  
- $\text{MSE}(w_R) = 0.370075$  

**Ridge performs slightly better.**

---

# Summary Table

| Method | Result |
|--------|--------|
| Bernoulli EM | Best clustering model |
| Gaussian EM | Poor fit for binary data |
| K-Means | Fast but limited |
| GD | Smooth convergence |
| SGD | Faster updates, small noise |
| Ridge | Best generalization |
