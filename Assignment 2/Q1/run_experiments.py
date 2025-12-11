import numpy as np
import matplotlib.pyplot as plt
import os
import time

from bernoulli_em import bernoulli_em
from gmm_em_full import gaussian_em_full
from kmeans import kmeans
from model_selection_metrics import (
        count_params_bernoulli,
        count_params_gaussian,
        compute_aic_bic
    )


def plot_mean_std_curve(mean, std, label, ylabel, title, outpath):
    iters = np.arange(1, len(mean) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(iters, mean, lw=2, label=label)
    plt.fill_between(iters, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.close()


def main():
    
    data_file = "A2Q1.csv"
    K = 4
    max_iter = 100
    n_inits = 100
    outdir = "assignment_outputs_final"
    os.makedirs(outdir, exist_ok=True)

   
    X = np.loadtxt(data_file, delimiter=",")
    if X.shape[0] < X.shape[1]:
        X = X.T
    N, D = X.shape
    print(f"Data shape: {X.shape}")

    
    ber_lls = np.zeros((n_inits, max_iter))
    gmm_lls = np.zeros((n_inits, max_iter))
    km_objs = np.zeros((n_inits, max_iter))
    mu_list = []

    start = time.time()
    for i in range(n_inits):
        rng = np.random.RandomState(i + 123)

        b_hist, _, mu = bernoulli_em(X, K, max_iter=max_iter, rng=rng)
        g_hist, _, _, _ = gaussian_em_full(X, K, max_iter=max_iter, rng=rng)
        k_hist, _, _ = kmeans(X, K, max_iter=max_iter, rng=rng)

        ber_lls[i, :] = b_hist
        gmm_lls[i, :] = g_hist
        km_objs[i, :] = k_hist
        mu_list.append(mu)

        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n_inits} runs "
                  f"(elapsed {time.time() - start:.1f}s)")

 
    mean_ber, std_ber = ber_lls.mean(0), ber_lls.std(0)
    mean_gmm, std_gmm = gmm_lls.mean(0), gmm_lls.std(0)
    mean_km, std_km = km_objs.mean(0), km_objs.std(0)

   


 
    plot_mean_std_curve(mean_ber, std_ber, "Bernoulli EM",
                        "Average Log-Likelihood",
                        f"Bernoulli Mixture EM (K={K}, avg over {n_inits} inits)",
                        os.path.join(outdir, "bernoulli_em_avg_ll.png"))

    plot_mean_std_curve(mean_gmm, std_gmm, "Gaussian EM (full)",
                        "Average Log-Likelihood",
                        f"Gaussian Mixture EM (K={K}, avg over {n_inits} inits)",
                        os.path.join(outdir, "gmm_em_avg_ll.png"))

    plot_mean_std_curve(mean_km, std_km, "K-Means objective",
                        "Average Objective (Sum of Squares)",
                        f"K-Means (K={K}, avg over {n_inits} inits)",
                        os.path.join(outdir, "kmeans_avg_obj.png"))

    
    best_idx = np.argmax([ll[-1] for ll in ber_lls])
    best_mu = mu_list[best_idx]

    print(f"\nBest Bernoulli EM run index: {best_idx} "
          f"(final LL = {ber_lls[best_idx, -1]:.3f})")

    comp_dir = os.path.join(outdir, "bernoulli_components")
    os.makedirs(comp_dir, exist_ok=True)

    for k in range(K):
        plt.figure(figsize=(3, 3))
        plt.imshow(best_mu[k].reshape(10, 5), cmap="gray",
                   vmin=0, vmax=1, interpolation="nearest")
        plt.title(f"Component {k+1}")
        plt.colorbar(label="P(x=1)")
        plt.tight_layout()
        save_path = os.path.join(comp_dir, f"component_{k+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"\nComponent images saved in {comp_dir}")
    print("All results saved in:", outdir)
    print("Final averages -> "
          f"Bernoulli LL {mean_ber[-1]:.3f}, "
          f"Gaussian LL {mean_gmm[-1]:.3f}, "
          f"KMeans Obj {mean_km[-1]:.3f}")

    
    final_bern_ll = mean_ber[-1]
    final_gmm_ll = mean_gmm[-1]
    N, D = X.shape

    p_bern = count_params_bernoulli(K, D)
    p_gmm = count_params_gaussian(K, D)

    aic_bern, bic_bern = compute_aic_bic(final_bern_ll, p_bern, N)
    aic_gmm, bic_gmm = compute_aic_bic(final_gmm_ll, p_gmm, N)

    print("\n===== Model Selection (AIC / BIC) =====")
    print(f"{'Model':<15}{'Log-Likelihood':>20}{'AIC':>20}{'BIC':>20}")
    print("-" * 70)
    print(f"{'Bernoulli EM':<15}{final_bern_ll:>20.2f}{aic_bern:>20.2f}{bic_bern:>20.2f}")
    print(f"{'Gaussian EM':<15}{final_gmm_ll:>20.2f}{aic_gmm:>20.2f}{bic_gmm:>20.2f}")
    print("-" * 70)

    better_aic = "Bernoulli" if aic_bern < aic_gmm else "Gaussian"
    better_bic = "Bernoulli" if bic_bern < bic_gmm else "Gaussian"
    print(f"\n→ Lower AIC suggests: {better_aic}")
    print(f"→ Lower BIC suggests: {better_bic}")
    print("\nNote: K-Means is not probabilistic; AIC/BIC are not applicable.")


    models = ["Bernoulli EM", "Gaussian EM"]
    aic_values = [aic_bern, aic_gmm]
    bic_values = [bic_bern, bic_gmm]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, aic_values, width, label="AIC")
    plt.bar(x + width/2, bic_values, width, label="BIC")
    plt.xticks(x, models)
    plt.ylabel("Metric Value")
    plt.title("Model Comparison: AIC and BIC (Lower is Better)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    aicbic_path = os.path.join(outdir, "AIC_BIC_comparison.png")
    plt.savefig(aicbic_path, dpi=300)
    plt.show()

    print(f"\nAIC/BIC comparison plot saved to: {aicbic_path}")

if __name__ == "__main__":
    main()
