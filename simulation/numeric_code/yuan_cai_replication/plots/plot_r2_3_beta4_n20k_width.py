"""Plot CI width only for n=20000, r2=3, beta~k^{-4}, excluding FPCA."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    K = 200
    n_val = 20000
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    C_a, C_i = 0.005, 0.005
    fname = f'alpha_sweep_aligned_r2_3_beta4_n{n_val}_Ca{C_a}_Ci{C_i}_v13.npz'
    data = np.load(os.path.join(results_dir, fname), allow_pickle=True)
    all_yhat = data['all_yhat']
    all_se = data['all_se']
    true_vals = data['true_vals']
    alphas = data['alphas']
    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]

    # Methods to plot (skip m=2 FPCA)
    methods = [
        (0, r'Aniso (Lepski $\kappa$)', 'tab:orange', 's'),
        (1, r'Iso (Lepski $\kappa$)', 'tab:green', 'D'),
        (3, r'Trunc.Aniso $5 n^{0.3}$', 'tab:blue', '^'),
        (4, r'Trunc.Aniso $5 n^{0.4}$', 'tab:red', 'o'),
        (5, r'Adapt.Trunc $5 n^{0.2}$', 'tab:cyan', 'P'),
    ]
    n_m = len(methods)

    # Compute widths
    width_mean = {}  # (m, ci) -> (mean, std) across test points
    for m, _, _, _ in methods:
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m in (3,):
                J = min(int(5 * n_val ** 0.3), K)
                q_two = t_dist.ppf(1 - alpha / 2, n_val - J)
            elif m == 4:
                J = min(int(5 * n_val ** 0.4), K)
                q_two = t_dist.ppf(1 - alpha / 2, n_val - J)
            elif m == 5:
                J = min(int(5 * n_val ** 0.2), K)
                q_two = t_dist.ppf(1 - alpha / 2, n_val - J)
            else:
                q_two = norm.ppf(1 - alpha / 2)

            w = 2 * q_two * all_se[m]  # (n_datasets, n_test)
            w_per_test = w.mean(axis=0)  # mean over datasets per test point
            width_mean[(m, si)] = (w_per_test.mean(), w_per_test.std())

    # Plot
    jitter = np.linspace(-0.18, 0.18, n_m)
    x_base = np.arange(n_conf)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mi, (m, label, color, marker) in enumerate(methods):
        x_vals, y_means, y_stds = [], [], []
        for ci in range(n_conf):
            ym, ys = width_mean[(m, ci)]
            x_vals.append(x_base[ci] + jitter[mi])
            y_means.append(ym)
            y_stds.append(ys)
        ax.errorbar(x_vals, y_means, yerr=y_stds,
                    color=color, marker=marker, markersize=8,
                    linewidth=0, elinewidth=1.5, capsize=4, capthick=1.2,
                    label=label, alpha=0.9)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f'{cl:.0%}' for cl in conf_levels], fontsize=11)
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Mean CI Width', fontsize=11)

    J3 = min(int(5 * n_val ** 0.3), K)
    J4 = min(int(5 * n_val ** 0.4), K)
    J2 = min(int(5 * n_val ** 0.2), K)
    ax.set_title(rf'CI Width — $r_2=3$, $b_k \sim k^{{-4}}$ (C1 satisfied), $n={n_val}$'
                 f'\n$J_{{0.3}}={J3},\\ J_{{0.4}}={J4},\\ J_{{0.2}}={J2}$',
                 fontsize=11)
    ax.legend(fontsize=9, loc='best')
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(max(0, y_lo), y_hi)

    plt.tight_layout()
    out_name = f'width_r2_3_beta4_n{n_val}_noFPCA.pdf'
    plt.savefig(os.path.join(results_dir, out_name),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {out_name}')
