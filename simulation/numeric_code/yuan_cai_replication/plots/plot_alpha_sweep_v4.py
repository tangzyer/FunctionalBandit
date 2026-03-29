"""Plot alpha sweep at 3 confidence levels × 3 sample sizes, jittered error bars."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


def load_and_compute(npz_path, n_val, target_conf_levels=(0.75, 0.85, 0.95)):
    """Load data and compute per-test-function coverage/width at selected levels."""
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']
    all_se = data['all_se']
    true_vals = data['true_vals']
    alphas = data['alphas']

    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)

    cov_per_test = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 4:
                J = int(np.sqrt(n_val))
                df = n_val - J
                q = t_dist.ppf(1 - alpha / 2, df)
            else:
                q = norm.ppf(1 - alpha / 2)

            lower = all_yhat[m] - q * all_se[m]
            upper = all_yhat[m] + q * all_se[m]
            width = upper - lower
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            cov_per_test[m, si, :] = covered.mean(axis=0)
            width_per_test[m, si, :] = width.mean(axis=0)

    return cov_per_test, width_per_test


def plot_combined(save_path):
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # Indices into the 5-method arrays: 1=aniso lam/(logn)^2, 3=iso lam/(logn)^2, 4=FPCA
    method_indices = [1, 3, 4]
    method_names = [
        r'Aniso $\lambda/(\log n)^2$',
        r'Iso $\lambda/(\log n)^2$',
        r'FPCA $\sqrt{n}$',
    ]
    colors = ['tab:orange', 'tab:green', 'tab:purple']
    markers = ['s', 'D', 'v']
    n_methods = len(method_indices)

    case_info = [
        ('aligned', r'Aligned ($r_2=1.5$)'),
        ('shifted', r'Shifted $k_0=10$'),
        ('haar', r'Haar ($r_2=1.5$)'),
    ]
    n_values = [64, 256, 1024]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    # Load all data: [case][n_idx] -> (cov_per_test, width_per_test)
    all_data = {}
    for case_name, _ in case_info:
        all_data[case_name] = {}
        for n_val in n_values:
            if n_val == 256:
                fname = f'alpha_sweep_{case_name}.npz'
            else:
                fname = f'alpha_sweep_{case_name}_n{n_val}.npz'
            path = os.path.join(results_dir, fname)
            cov_pt, width_pt = load_and_compute(path, n_val)
            all_data[case_name][n_val] = (cov_pt, width_pt)

    # Layout: 2 rows (coverage, width) × 3 columns (cases)
    # x-axis: 9 groups = 3 n-values × 3 conf-levels
    # Within each group: 5 jittered methods

    # x positions: group by n, sub-group by confidence level
    # n=64:   x = 1, 2, 3
    # n=256:  x = 5, 6, 7
    # n=1024: x = 9, 10, 11
    group_starts = {64: 1, 256: 5, 1024: 9}
    x_positions = {}
    for ni, n_val in enumerate(n_values):
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    # Method jitter
    jitter = np.linspace(-0.2, 0.2, n_methods)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, (case_name, case_label) in enumerate(case_info):
        for row, (metric_name, ylabel) in enumerate([
            ('coverage', 'Actual Coverage'),
            ('width', 'Mean CI Width'),
        ]):
            ax = axes[row, col]

            # Collect all data points for y-limits
            all_y_lo, all_y_hi = [], []

            for mi, m_data_idx in enumerate(method_indices):
                x_vals = []
                y_means = []
                y_stds = []

                for n_val in n_values:
                    cov_pt, width_pt = all_data[case_name][n_val]
                    if metric_name == 'coverage':
                        pt = cov_pt  # (5, n_conf, n_test)
                    else:
                        pt = width_pt

                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)] + jitter[mi]
                        ym = pt[m_data_idx, ci, :].mean()
                        ys = pt[m_data_idx, ci, :].std()
                        x_vals.append(xp)
                        y_means.append(ym)
                        y_stds.append(ys)
                        all_y_lo.append(ym - ys)
                        all_y_hi.append(ym + ys)

                ax.errorbar(x_vals, y_means, yerr=y_stds,
                            color=colors[mi], marker=markers[mi], markersize=6,
                            linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                            label=method_names[mi] if (col == 0 and row == 0) else None,
                            alpha=0.9)

            # Ideal coverage line (scatter at each group position)
            if metric_name == 'coverage':
                for n_val in n_values:
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)]
                        ax.scatter([xp], [conf_levels[ci]], marker='x', color='black',
                                   s=60, zorder=10, linewidths=1.5,
                                   label='Ideal' if (n_val == 64 and ci == 0 and col == 0) else None)
                        all_y_lo.append(conf_levels[ci])
                        all_y_hi.append(conf_levels[ci])

            # y-axis limits
            y_lo = min(all_y_lo)
            y_hi = max(all_y_hi)
            pad = (y_hi - y_lo) * 0.06
            ax.set_ylim(max(0, y_lo - pad), y_hi + pad)

            # x-axis labels
            tick_positions = []
            tick_labels = []
            for n_val in n_values:
                for ci in range(n_conf):
                    tick_positions.append(x_positions[(n_val, ci)])
                    tick_labels.append(f'{conf_levels[ci]:.0%}')

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)

            # Add n labels below
            for n_val in n_values:
                center = group_starts[n_val] + 1  # middle of the 3 conf levels
                ax.text(center, -0.13, f'$n={n_val}$', transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=10, fontweight='bold')

            # Vertical separators between n-groups
            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

            ax.set_ylabel(ylabel)
            ax.set_title(f'{"CI Width" if row == 1 else "Coverage"} — {case_label}')

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(r'Coverage & CI Width at 75%, 85%, 95% Confidence, $n \in \{64, 256, 1024\}$',
                 fontsize=14, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved plot to {save_path}')


if __name__ == '__main__':
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_combined(os.path.join(results_dir, 'inference_alpha_sweep_v4.pdf'))
