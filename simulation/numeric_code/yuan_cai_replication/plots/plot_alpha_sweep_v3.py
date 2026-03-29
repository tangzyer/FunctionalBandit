"""Plot alpha sweep at 3 selected confidence levels with jittered error bars."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, t as t_dist


def load_and_compute(npz_path, target_conf_levels=(0.75, 0.85, 0.95)):
    """Load data and compute coverage/width at selected confidence levels."""
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']    # (n_methods, n_datasets, n_test)
    all_se = data['all_se']        # (n_methods, n_datasets, n_test)
    true_vals = data['true_vals']  # (n_test,)
    alphas = data['alphas']        # (30,)

    n_methods, n_datasets, n_test = all_yhat.shape

    # Find indices closest to target confidence levels
    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = []
    for ta in target_alphas:
        idx = np.argmin(np.abs(alphas - ta))
        indices.append(idx)

    n_sel = len(indices)
    cov_per_test = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 4:  # FPCA uses t-distribution
                n_sample = 256
                J = int(np.sqrt(n_sample))
                df = n_sample - J
                q = t_dist.ppf(1 - alpha / 2, df)
            else:
                q = norm.ppf(1 - alpha / 2)

            lower = all_yhat[m] - q * all_se[m]
            upper = all_yhat[m] + q * all_se[m]
            width = upper - lower

            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            cov_per_test[m, si, :] = covered.mean(axis=0)
            width_per_test[m, si, :] = width.mean(axis=0)

    actual_conf_levels = np.array([1 - alphas[i] for i in indices])
    return actual_conf_levels, cov_per_test, width_per_test


def plot_alpha_sweep_sparse(case_data, case_labels, save_path):
    method_names = [
        r'Aniso oracle $\lambda$',
        r'Aniso $\lambda/(\log n)^2$',
        r'Iso oracle $\lambda$',
        r'Iso $\lambda/(\log n)^2$',
        r'FPCA $\sqrt{n}$',
    ]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    n_methods = 5

    # Horizontal jitter offsets for each method
    jitter = np.linspace(-0.015, 0.015, n_methods)

    n_cases = len(case_data)
    fig, axes = plt.subplots(2, n_cases, figsize=(5.5 * n_cases, 9))
    if n_cases == 1:
        axes = axes[:, None]

    for col, (conf_levels, cov_pt, width_pt) in enumerate(case_data):
        n_sel = len(conf_levels)

        cov_mean = cov_pt.mean(axis=2)   # (n_methods, n_sel)
        cov_std = cov_pt.std(axis=2)
        width_mean = width_pt.mean(axis=2)
        width_std = width_pt.std(axis=2)

        # --- Coverage panel ---
        ax = axes[0, col]
        # Ideal markers
        ax.scatter(conf_levels, conf_levels, marker='x', color='black',
                   s=80, zorder=10, linewidths=1.5, label='Ideal')

        for m in range(n_methods):
            x_jittered = conf_levels + jitter[m]
            ax.errorbar(x_jittered, cov_mean[m], yerr=cov_std[m],
                        color=colors[m], marker=markers[m], markersize=7,
                        linewidth=0, elinewidth=1.5, capsize=4, capthick=1.2,
                        label=method_names[m], alpha=0.9)

        # Tight y-axis
        all_lo = [cov_mean[m] - cov_std[m] for m in range(n_methods)]
        all_hi = [cov_mean[m] + cov_std[m] for m in range(n_methods)]
        y_lo = min(np.min(a) for a in all_lo)
        y_hi = max(np.max(a) for a in all_hi)
        # Also include ideal points
        y_lo = min(y_lo, conf_levels.min())
        y_hi = max(y_hi, conf_levels.max())
        pad = (y_hi - y_lo) * 0.08
        ax.set_ylim(max(0, y_lo - pad), min(1.05, y_hi + pad))

        ax.set_xticks(conf_levels)
        ax.set_xticklabels([f'{cl:.2f}' for cl in conf_levels])
        ax.set_xlabel(r'Confidence Level $(1-\alpha)$')
        ax.set_ylabel('Actual Coverage')
        ax.set_title(f'Coverage — {case_labels[col]}')
        if col == 0:
            ax.legend(fontsize=7.5, loc='best')

        # --- CI Width panel ---
        ax = axes[1, col]
        for m in range(n_methods):
            x_jittered = conf_levels + jitter[m]
            ax.errorbar(x_jittered, width_mean[m], yerr=width_std[m],
                        color=colors[m], marker=markers[m], markersize=7,
                        linewidth=0, elinewidth=1.5, capsize=4, capthick=1.2,
                        label=method_names[m], alpha=0.9)

        all_lo_w = [width_mean[m] - width_std[m] for m in range(n_methods)]
        all_hi_w = [width_mean[m] + width_std[m] for m in range(n_methods)]
        y_lo_w = min(np.min(a) for a in all_lo_w)
        y_hi_w = max(np.max(a) for a in all_hi_w)
        pad_w = (y_hi_w - y_lo_w) * 0.08
        ax.set_ylim(max(0, y_lo_w - pad_w), y_hi_w + pad_w)

        ax.set_xticks(conf_levels)
        ax.set_xticklabels([f'{cl:.2f}' for cl in conf_levels])
        ax.set_xlabel(r'Confidence Level $(1-\alpha)$')
        ax.set_ylabel('Mean CI Width')
        ax.set_title(f'CI Width — {case_labels[col]}')
        if col == 0:
            ax.legend(fontsize=7.5, loc='best')

    fig.suptitle(r'Coverage & CI Width at Selected Confidence Levels, $n = 256$',
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved plot to {save_path}')


if __name__ == '__main__':
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    cases = []
    labels = []

    for fname, label in [
        ('alpha_sweep_aligned.npz', r'Aligned ($r_2=1.5$)'),
        ('alpha_sweep_shifted.npz', r'Shifted $k_0=10$'),
        ('alpha_sweep_haar.npz', r'Haar ($r_2=1.5$)'),
    ]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            cases.append(load_and_compute(path))
            labels.append(label)

    if cases:
        save_path = os.path.join(results_dir, 'inference_alpha_sweep_v3.pdf')
        plot_alpha_sweep_sparse(cases, labels, save_path)
    else:
        print('No alpha sweep data found!')
