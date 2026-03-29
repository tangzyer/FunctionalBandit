"""Plot one-sided coverage + two-sided CI width for aligned r2=2.

3 rows: left coverage, right coverage, CI width.
Lambda grid 10^-4 to 1, excess-risk tuned.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


def load_and_compute(npz_path, n_val, target_conf_levels=(0.75, 0.85, 0.95)):
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']    # (n_methods, n_datasets, n_test)
    all_se = data['all_se']
    true_vals = data['true_vals']
    alphas = data['alphas']
    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)

    left_cov = np.zeros((n_methods, n_sel, n_test))
    right_cov = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 4:
                J = int(np.sqrt(n_val))
                df = n_val - J
                q_one = t_dist.ppf(1 - alpha, df)
                q_two = t_dist.ppf(1 - alpha / 2, df)
            else:
                q_one = norm.ppf(1 - alpha)
                q_two = norm.ppf(1 - alpha / 2)

            # One-sided
            upper_one = all_yhat[m] + q_one * all_se[m]
            lower_one = all_yhat[m] - q_one * all_se[m]
            left_cov[m, si, :] = (true_vals[None, :] <= upper_one).mean(axis=0)
            right_cov[m, si, :] = (true_vals[None, :] >= lower_one).mean(axis=0)

            # Two-sided width
            width = 2 * q_two * all_se[m]  # (n_datasets, n_test)
            width_per_test[m, si, :] = width.mean(axis=0)

    return left_cov, right_cov, width_per_test


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    method_indices = [1, 3, 4]
    method_names = [
        r'Aniso $\lambda/((\log n)^3\log\log n)$',
        r'Iso $\lambda/((\log n)^2\log\log n)$',
        r'FPCA $\sqrt{n}$',
    ]
    colors = ['tab:orange', 'tab:green', 'tab:purple']
    markers = ['s', 'D', 'v']
    n_methods = len(method_indices)

    n_values = [64, 256, 1024]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    # Load data
    all_data = {}
    for n_val in n_values:
        fname = f'alpha_sweep_aligned_r2_2_n{n_val}_v5.npz'
        path = os.path.join(results_dir, fname)
        left_cov, right_cov, width_pt = load_and_compute(path, n_val)
        all_data[n_val] = (left_cov, right_cov, width_pt)

    group_starts = {64: 1, 256: 5, 1024: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    jitter = np.linspace(-0.2, 0.2, n_methods)

    row_info = [
        ('left', 'Left-sided Coverage', 'Actual Coverage'),
        ('right', 'Right-sided Coverage', 'Actual Coverage'),
        ('width', 'Two-sided CI Width', 'Mean CI Width'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for row, (metric, title, ylabel) in enumerate(row_info):
        ax = axes[row]
        all_y_lo, all_y_hi = [], []

        for mi, m_idx in enumerate(method_indices):
            x_vals, y_means, y_stds = [], [], []
            for n_val in n_values:
                left_cov, right_cov, width_pt = all_data[n_val]
                if metric == 'left':
                    pt = left_cov
                elif metric == 'right':
                    pt = right_cov
                else:
                    pt = width_pt
                for ci in range(n_conf):
                    xp = x_positions[(n_val, ci)] + jitter[mi]
                    ym = pt[m_idx, ci, :].mean()
                    ys = pt[m_idx, ci, :].std()
                    x_vals.append(xp)
                    y_means.append(ym)
                    y_stds.append(ys)
                    all_y_lo.append(ym - ys)
                    all_y_hi.append(ym + ys)

            ax.errorbar(x_vals, y_means, yerr=y_stds,
                        color=colors[mi], marker=markers[mi], markersize=6,
                        linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                        label=method_names[mi], alpha=0.9)

        # Ideal markers for coverage rows
        if metric in ('left', 'right'):
            for n_val in n_values:
                for ci in range(n_conf):
                    xp = x_positions[(n_val, ci)]
                    ax.scatter([xp], [conf_levels[ci]], marker='x',
                               color='black', s=60, zorder=10, linewidths=1.5,
                               label='Ideal' if (n_val == 64 and ci == 0) else None)
                    all_y_lo.append(conf_levels[ci])
                    all_y_hi.append(conf_levels[ci])

        y_lo = min(all_y_lo)
        y_hi = max(all_y_hi)
        pad = (y_hi - y_lo) * 0.06
        if metric in ('left', 'right'):
            ax.set_ylim(max(0, y_lo - pad), min(1.02, y_hi + pad))
        else:
            ax.set_ylim(max(0, y_lo - pad), y_hi + pad)

        tick_positions, tick_labels_list = [], []
        for n_val in n_values:
            for ci in range(n_conf):
                tick_positions.append(x_positions[(n_val, ci)])
                tick_labels_list.append(f'{conf_levels[ci]:.0%}')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=9)

        for n_val in n_values:
            center = group_starts[n_val] + 1
            ax.text(center, -0.10, f'$n={n_val}$',
                    transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=10, fontweight='bold')

        for sep_x in [4, 8]:
            ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(r'Aligned ($r_2=2$), $\lambda$ grid $[10^{-5}, 1]$, excess-risk tuned, 1000 test fns',
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(results_dir, 'inference_alpha_sweep_v8_aligned.pdf'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved plot')
