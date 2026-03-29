"""Width-only plots (no FPCA) for the two C1-satisfied cases.
Row 1: beta~k^{-4},  Row 2: sparse beta=(4,-2,0,...).
3 columns: Aligned r2=3, Haar r2=3, Shifted k0=50."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


def load_and_compute(npz_path, n_val, target_conf_levels=(0.75, 0.85, 0.95)):
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']
    all_se = data['all_se']
    true_vals = data['true_vals']
    alphas = data['alphas']
    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)

    width_per_test = np.zeros((n_methods, n_sel, n_test))

    K = 200
    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 2:
                J = int(np.sqrt(n_val))
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 3:
                J = min(int(n_val ** 0.3), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 4:
                J = min(int(n_val ** 0.4), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 5:
                J = min(int(n_val ** 0.2), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            else:
                q_two = norm.ppf(1 - alpha / 2)

            width = 2 * q_two * all_se[m]
            width_per_test[m, si, :] = width.mean(axis=0)

    return width_per_test


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    K = 200

    # Two rows: beta4 and sparse2
    row_info = [
        (
            [('aligned_r2_3_beta4', r'Aligned ($r_2=3$)'),
             ('haar_r2_3_beta4', r'Haar ($r_2=3$)'),
             ('shifted_beta4', r'Shifted ($k_0=50$)')],
            r'$b_k \sim k^{-4}$'
        ),
        (
            [('aligned_r2_3_sparse2', r'Aligned ($r_2=3$)'),
             ('haar_r2_3_sparse2', r'Haar ($r_2=3$)'),
             ('shifted_sparse2', r'Shifted ($k_0=50$)')],
            r'Sparse $\beta=(4,-2,0,\ldots)$'
        ),
    ]

    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    # Methods to plot (exclude m=2 FPCA)
    plot_methods = [0, 1, 3, 4, 5]
    method_names = {
        0: r'Aniso (Lepski $\kappa$)',
        1: r'Iso (Lepski $\kappa$)',
        3: r'Trunc.Aniso $n^{0.3}$',
        4: r'Trunc.Aniso $n^{0.4}$',
        5: r'Adapt.Trunc $n^{0.2}$',
    }
    colors = {0: 'tab:orange', 1: 'tab:green', 3: 'tab:blue', 4: 'tab:red',
              5: 'tab:cyan'}
    markers = {0: 's', 1: 'D', 3: '^', 4: 'o', 5: 'P'}
    n_pm = len(plot_methods)
    jitter = np.linspace(-0.24, 0.24, n_pm)

    group_starts = {1000: 1, 2000: 5, 4000: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    C_a, C_i = 0.005, 0.005

    # Load all data
    all_data = {}
    for row_cases, _ in row_info:
        for case_name, _ in row_cases:
            all_data[case_name] = {}
            for n_val in n_values:
                fname = f'alpha_sweep_{case_name}_n{n_val}_Ca{C_a}_Ci{C_i}_v13.npz'
                path = os.path.join(results_dir, fname)
                wid = load_and_compute(path, n_val)
                all_data[case_name][n_val] = wid

    n_rows = len(row_info)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8))

    for row_idx, (row_cases, row_label) in enumerate(row_info):
        for col, (case_name, case_label) in enumerate(row_cases):
            ax = axes[row_idx, col]
            all_y_lo, all_y_hi = [], []
            for pi, mi in enumerate(plot_methods):
                x_vals, y_means, y_stds = [], [], []
                for n_val in n_values:
                    wid = all_data[case_name][n_val]
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)] + jitter[pi]
                        ym = wid[mi, ci, :].mean()
                        ys = wid[mi, ci, :].std()
                        x_vals.append(xp)
                        y_means.append(ym)
                        y_stds.append(ys)
                        all_y_lo.append(ym - ys)
                        all_y_hi.append(ym + ys)
                if col == 0 and row_idx == 0:
                    label = method_names[mi]
                else:
                    label = None
                ax.errorbar(x_vals, y_means, yerr=y_stds,
                            color=colors[mi], marker=markers[mi], markersize=6,
                            linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                            label=label, alpha=0.9)
            y_lo, y_hi = min(all_y_lo), max(all_y_hi)
            pad = (y_hi - y_lo) * 0.06
            ax.set_ylim(max(0, y_lo - pad), y_hi + pad)

            tick_positions, tick_labels_list = [], []
            for n_val in n_values:
                for ci in range(n_conf):
                    tick_positions.append(x_positions[(n_val, ci)])
                    tick_labels_list.append(f'{conf_levels[ci]:.0%}')
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels_list, fontsize=8)
            for n_val in n_values:
                center = group_starts[n_val] + 1
                J3 = min(int(n_val ** 0.3), K)
                J4 = min(int(n_val ** 0.4), K)
                J2 = min(int(n_val ** 0.2), K)
                ax.text(center, -0.16,
                        f'$n={n_val}$\n$J_{{0.3}}={J3}, J_{{0.4}}={J4}, J_{{0.2}}={J2}$',
                        transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=7, fontweight='bold')
            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.set_ylabel('Mean CI Width')
            ax.set_title(f'{row_label} — {case_label}')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(r'CI Width (excl. FPCA), C1 satisfied, $J=n^q$',
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_name = 'width_noFPCA_C1_3col.pdf'
    plt.savefig(os.path.join(results_dir, out_name),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {out_name}')
