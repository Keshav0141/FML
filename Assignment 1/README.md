# Foundations of Machine Learning – Assignment 1  
### PCA, Kernel PCA & Clustering Methods  
**Author:** Aryan Prasad  
**Roll Number:** DA25M007  

This repository contains code and analysis for two major machine learning tasks:  
1. Principal Component Analysis (PCA) & Kernel PCA  
2. Clustering on a synthetic concentric-rings dataset using K-Means, Voronoi diagrams, and Spectral Methods  

Source files:  
- `FML_Assign1_PCA_final.py`  
- `FML_Assign1_Kmeans_final.py`  
- `Report.pdf`  
- Provided datasets for PCA and clustering  

---

## 1. PCA & Kernel PCA  

### 1.1 Dataset  
The PCA dataset contains 1000 points in 2D forming spiral-like curves. The goal is to compare linear PCA with nonlinear Kernel PCA.

### 1.2 PCA  
Steps:  
- Center data  
- Compute covariance matrix  
- Perform eigen decomposition  
- Sort eigenvalues  
- Project onto principal components  

**Results:**  
- Eigenvalues: λ₁ ≈ 1.63, λ₂ ≈ 0.86  
- Explained variance: PC1 ≈ 65%, PC2 ≈ 35%  
PCA identifies main variance directions but cannot unwrap nonlinear spiral structures.

### 1.3 Kernel PCA  
Kernels: Linear, Polynomial (degree 3), RBF.  

**Observations:**  
- Linear kernel ≈ PCA  
- Polynomial kernel captures curvature but unstable  
- RBF kernel provides the best nonlinear separation  

**Best Kernel:** RBF with σ ≈ 1.0–2.0.

---

## 2. Clustering on Concentric Rings  

The dataset consists of four concentric rings. This challenges traditional clustering methods due to non-convex cluster shapes.

Methods used:  
- K-Means  
- Voronoi region visualization  
- Spectral clustering (Kernel PCA and Laplacian)  
- Alternative eigenvector-based mapping  

---

## 2.1 K-Means with Multiple Seeds  

Tested with seeds: 0, 7, 21, 42, 99.  

**Observations:**  
- Inertia drops quickly → fast convergence  
- Different seeds → different clusters  
- K-Means partitions form radial wedges, slicing rings  
- K-Means fails because Voronoi boundaries are convex and cannot model curved structures  

---

## 2.2 Voronoi Regions (K = 2, 3, 4, 5)  

**Findings:**  
- K = 2, 3 merge rings → underfitting  
- K = 4 matches ring count but still slices rings  
- K = 5 over-splits  

Conclusion: Voronoi boundaries confirm limitations of K-Means for circular structures.

---

## 2.3 Spectral Clustering  

Similarity defined using Gaussian kernel:

$$
W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
$$


Two embeddings used:  
1. Kernel PCA-style  
2. Normalized Laplacian  

**Results:**  
- σ = 0.1 → unstable  
- σ = 0.3 → Laplacian clustering perfectly separates all four rings  
- σ ≥ 0.5 → rings merge  

**Best Method:** Laplacian spectral clustering with σ ≈ 0.3.

---

## 2.4 Alternative Mapping (argmax on eigenvectors)  

Assignment rule:

$$
\ell(i) = \arg\max_{1 \le j \le k} \; v_{ji}
$$
 

**Observations:**  
- Kernel method collapses to 1–2 clusters  
- Laplacian somewhat better but noisy  
- Worse than spectral embedding + K-Means  

Conclusion: direct eigenvector argmax mapping is unreliable.

---

## Final Insights  

- PCA captures variance directions but fails on nonlinear shapes  
- Kernel PCA (especially RBF) effectively unfolds nonlinear structures  
- K-Means is unsuitable for ring-shaped clusters  
- Voronoi diagrams reveal geometric rigidity of K-Means  
- Laplacian spectral clustering (σ = 0.3) consistently recovers all rings  
- Eigenvector argmax mapping is inferior to spectral embedding + K-Means  

