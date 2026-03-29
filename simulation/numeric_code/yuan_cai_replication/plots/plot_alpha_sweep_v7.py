"""Plot one-sided coverage for prediction-risk tuned lambda.

Same layout as v6 but using v4 data (prediction-risk tuning per test fn).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


def load_and_compute_onesided(npz_path, n_val, target_conf_levels=(0.75, 0.85, 0.95)):
    """Compute left and right one-sided coverage per test function."""
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']    # (n_methods, n_datasets, n_test)
    all_se = data['all_se']
    true_vals = data['true_vals']  # (n_test,)
    alphas = data['alphas']
    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)

    left_cov = np.zeros((n_methods, n_sel, n_test))
    right_cov = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 4:  # FPCA uses t-distribution
                J = int(np.sqrt(n_val))
                df = n_val - J
                q = t_dist.ppf(1 - alpha, df)
            else:
                q = norm.ppf(1 - alpha)

            upper = all_yhat[m] + q * all_se[m]
            lower = all_yhat[m] - q * all_se[m]

            left_covered = (true_vals[None, :] <= upper)
            left_cov[m, si, :] = left_covered.mean(axis=0)

            right_covered = (true_vals[None, :] >= lower)
            right_cov[m, si, :] = right_covered.mean(axis=0)

    return left_cov, right_cov


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    method_indices = [1, 3, 4]
    method_names = [
        r'Aniso $\lambda/(\log n)^3$',
        r'Iso $\lambda/\log\log n$',
        r'FPCA $\sqrt{n}$',
    ]
    colors = ['tab:orange', 'tab:green', 'tab:purple']
    markers = ['s', 'D', 'v']
    n_methods = len(method_indices)

    case_info = [
        ('aligned_r2_2', r'Aligned ($r_2=2$)'),
        ('shifted', r'Shifted $k_0=10$'),
        ('haar_r2_2', r'Haar ($r_2=2$)'),
    ]
    n_values = [64, 256, 1024]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    # Load all data (v4 = prediction-risk tuned)
    all_data = {}
    for case_name, _ in case_info:
        all_data[case_name] = {}
        for n_val in n_values:
            fname = f'alpha_sweep_{case_name}_n{n_val}_v4.npz'
            path = os.path.join(results_dir, fname)
            left_cov, right_cov = load_and_compute_onesided(path, n_val)
            all_data[case_name][n_val] = (left_cov, right_cov)

    group_starts = {64: 1, 256: 5, 1024: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    jitter = np.linspace(-0.2, 0.2, n_methods)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, (case_name, case_label) in enumerate(case_info):
        for row, side_name in enumerate(['Left', 'Right']):
            ax = axes[row, col]
            all_y_lo, all_y_hi = [], []

            for mi, m_data_idx in enumerate(method_indices):
                x_vals, y_means, y_stds = [], [], []
                for n_val in n_values:
                    left_cov, right_cov = all_data[case_name][n_val]
                    pt = left_cov if side_name == 'Left' else right_cov
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

            # Ideal markers
            for n_val in n_values:
                for ci in range(n_conf):
                    xp = x_positions[(n_val, ci)]
                    ax.scatter([xp], [conf_levels[ci]], marker='x', color='black',
                               s=60, zorder=10, linewidths=1.5,
                               label='Ideal' if (n_val == 64 and ci == 0 and col == 0 and row == 0) else None)
                    all_y_lo.append(conf_levels[ci])
                    all_y_hi.append(conf_levels[ci])

            y_lo = min(all_y_lo)
            y_hi = max(all_y_hi)
            pad = (y_hi - y_lo) * 0.06
            ax.set_ylim(max(0, y_lo - pad), min(1.02, y_hi + pad))

            tick_positions, tick_labels_list = [], []
            for n_val in n_values:
                for ci in range(n_conf):
                    tick_positions.append(x_positions[(n_val, ci)])
                    tick_labels_list.append(f'{conf_levels[ci]:.0%}')
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels_list, fontsize=8)

            for n_val in n_values:
                center = group_starts[n_val] + 1
                ax.text(center, -0.13, f'$n={n_val}$', transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=10, fontweight='bold')

            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

            ax.set_ylabel('Actual Coverage')
            ax.set_title(f'{side_name}-sided — {case_label}')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(r'One-sided Coverage (Prediction-Risk Tuned $\lambda$), $r_2=2$, 1000 test fns',
                 fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(results_dir, 'inference_alpha_sweep_v7_predrisk.pdf'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved plot')
