"""Plot Lepski coverage/width with kappas 0.03, 0.1, 0.3, dense lambda grid (2000 pts).

Data from v10 files (7 methods: 3 aniso + 3 iso + FPCA).
"""

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

    two_sided_cov = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 6:  # FPCA
                J = int(np.sqrt(n_val))
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            else:
                q_two = norm.ppf(1 - alpha / 2)

            upper = all_yhat[m] + q_two * all_se[m]
            lower = all_yhat[m] - q_two * all_se[m]
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            two_sided_cov[m, si, :] = covered.mean(axis=0)

            width = 2 * q_two * all_se[m]
            width_per_test[m, si, :] = width.mean(axis=0)

    return two_sided_cov, width_per_test


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    case_info = [
        ('aligned_r2_2', r'Aligned ($r_2=2$)'),
        ('shifted', r'Shifted $k_0=10$'),
        ('haar_r2_2', r'Haar ($r_2=2$)'),
    ]
    n_values = [64, 256, 1024]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    all_data = {}
    for case_name, _ in case_info:
        all_data[case_name] = {}
        for n_val in n_values:
            fname = f'alpha_sweep_{case_name}_n{n_val}_v10.npz'
            path = os.path.join(results_dir, fname)
            two_sided_cov, width_pt = load_and_compute(path, n_val)
            all_data[case_name][n_val] = (two_sided_cov, width_pt)

    group_starts = {64: 1, 256: 5, 1024: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    row_info = [
        ('coverage', 'Two-sided Coverage', 'Actual Coverage'),
        ('width', 'Two-sided CI Width', 'Mean CI Width'),
    ]

    # --- Plot 1: Aniso kappa comparison ---
    method_indices_a = [0, 1, 2, 6]
    method_names_a = [
        r'Aniso $\kappa=0.03$',
        r'Aniso $\kappa=0.1$',
        r'Aniso $\kappa=0.3$',
        r'FPCA $\sqrt{n}$',
    ]
    colors_a = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
    markers_a = ['o', 's', '^', 'v']
    n_ma = len(method_indices_a)
    jitter_a = np.linspace(-0.25, 0.25, n_ma)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for col, (case_name, case_label) in enumerate(case_info):
        for row, (metric, row_title, ylabel) in enumerate(row_info):
            ax = axes[row, col]
            all_y_lo, all_y_hi = [], []
            for mi, m_idx in enumerate(method_indices_a):
                x_vals, y_means, y_stds = [], [], []
                for n_val in n_values:
                    two_sided_cov, width_pt = all_data[case_name][n_val]
                    pt = two_sided_cov if metric == 'coverage' else width_pt
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)] + jitter_a[mi]
                        ym = pt[m_idx, ci, :].mean()
                        ys = pt[m_idx, ci, :].std()
                        x_vals.append(xp)
                        y_means.append(ym)
                        y_stds.append(ys)
                        all_y_lo.append(ym - ys)
                        all_y_hi.append(ym + ys)
                ax.errorbar(x_vals, y_means, yerr=y_stds,
                            color=colors_a[mi], marker=markers_a[mi], markersize=6,
                            linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                            label=method_names_a[mi] if (col == 0 and row == 0) else None,
                            alpha=0.9)
            if metric == 'coverage':
                for n_val in n_values:
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)]
                        ax.scatter([xp], [conf_levels[ci]], marker='x',
                                   color='black', s=60, zorder=10, linewidths=1.5,
                                   label='Ideal' if (n_val == 64 and ci == 0 and col == 0 and row == 0) else None)
                        all_y_lo.append(conf_levels[ci])
                        all_y_hi.append(conf_levels[ci])
            y_lo, y_hi = min(all_y_lo), max(all_y_hi)
            pad = (y_hi - y_lo) * 0.06
            if metric == 'coverage':
                ax.set_ylim(max(0, y_lo - pad), min(1.02, y_hi + pad))
            else:
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
                ax.text(center, -0.13, f'$n={n_val}$',
                        transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=10, fontweight='bold')
            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{row_title} — {case_label}')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r'Aniso Lepski: $\kappa=0.03,0.1,0.3$, 2000-pt $\lambda$ grid',
                 fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(results_dir, 'inference_alpha_sweep_v44_aniso_kappa.pdf'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved plot v44 (Aniso)')

    # --- Plot 2: Iso kappa comparison ---
    method_indices_i = [3, 4, 5, 6]
    method_names_i = [
        r'Iso $\kappa=0.03$',
        r'Iso $\kappa=0.1$',
        r'Iso $\kappa=0.3$',
        r'FPCA $\sqrt{n}$',
    ]
    colors_i = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']
    markers_i = ['o', 'D', '^', 'v']
    n_mi = len(method_indices_i)
    jitter_i = np.linspace(-0.25, 0.25, n_mi)

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9))
    for col, (case_name, case_label) in enumerate(case_info):
        for row, (metric, row_title, ylabel) in enumerate(row_info):
            ax = axes2[row, col]
            all_y_lo, all_y_hi = [], []
            for mi, m_idx in enumerate(method_indices_i):
                x_vals, y_means, y_stds = [], [], []
                for n_val in n_values:
                    two_sided_cov, width_pt = all_data[case_name][n_val]
                    pt = two_sided_cov if metric == 'coverage' else width_pt
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)] + jitter_i[mi]
                        ym = pt[m_idx, ci, :].mean()
                        ys = pt[m_idx, ci, :].std()
                        x_vals.append(xp)
                        y_means.append(ym)
                        y_stds.append(ys)
                        all_y_lo.append(ym - ys)
                        all_y_hi.append(ym + ys)
                ax.errorbar(x_vals, y_means, yerr=y_stds,
                            color=colors_i[mi], marker=markers_i[mi], markersize=6,
                            linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                            label=method_names_i[mi] if (col == 0 and row == 0) else None,
                            alpha=0.9)
            if metric == 'coverage':
                for n_val in n_values:
                    for ci in range(n_conf):
                        xp = x_positions[(n_val, ci)]
                        ax.scatter([xp], [conf_levels[ci]], marker='x',
                                   color='black', s=60, zorder=10, linewidths=1.5,
                                   label='Ideal' if (n_val == 64 and ci == 0 and col == 0 and row == 0) else None)
                        all_y_lo.append(conf_levels[ci])
                        all_y_hi.append(conf_levels[ci])
            y_lo, y_hi = min(all_y_lo), max(all_y_hi)
            pad = (y_hi - y_lo) * 0.06
            if metric == 'coverage':
                ax.set_ylim(max(0, y_lo - pad), min(1.02, y_hi + pad))
            else:
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
                ax.text(center, -0.13, f'$n={n_val}$',
                        transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=10, fontweight='bold')
            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{row_title} — {case_label}')
    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='lower center', ncol=5, fontsize=9,
                bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle(r'Iso Lepski: $\kappa=0.03,0.1,0.3$, 2000-pt $\lambda$ grid',
                  fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(results_dir, 'inference_alpha_sweep_v44_iso_kappa.pdf'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved plot v44 (Iso)')
