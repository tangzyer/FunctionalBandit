"""Plot Lepski coverage/width for v13 — r2=3, sparse beta, 6 methods."""

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

    K = 200
    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 2:  # FPCA
                J = int(np.sqrt(n_val))
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 3:  # TruncAniso 5*n^0.3
                J = min(int(5 * n_val ** 0.3), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 4:  # TruncAniso 5*n^0.4
                J = min(int(5 * n_val ** 0.4), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 5:  # AdaptTrunc 5*n^0.2
                J = min(int(5 * n_val ** 0.2), K)
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            else:  # Lepski (m=0,1)
                q_two = norm.ppf(1 - alpha / 2)

            upper = all_yhat[m] + q_two * all_se[m]
            lower = all_yhat[m] - q_two * all_se[m]
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            two_sided_cov[m, si, :] = covered.mean(axis=0)

            width = 2 * q_two * all_se[m]
            width_per_test[m, si, :] = width.mean(axis=0)

    ka = float(data['kappa_aniso'])
    ki = float(data['kappa_iso'])

    return two_sided_cov, width_per_test, ka, ki


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    case_info = [
        ('aligned_r2_3_sparse', r'Aligned ($r_2=3$, sparse $\beta$)'),
    ]
    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    method_names = [
        r'Aniso (Lepski $\kappa$)',
        r'Iso (Lepski $\kappa$)',
        r'FPCA $\sqrt{n}$',
        r'Trunc.Aniso $5 n^{0.3}$',
        r'Trunc.Aniso $5 n^{0.4}$',
        r'Adapt.Trunc $5 n^{0.2}$',
    ]
    colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:blue', 'tab:red',
              'tab:cyan']
    markers = ['s', 'D', 'v', '^', 'o', 'P']
    n_m = len(method_names)
    jitter = np.linspace(-0.28, 0.28, n_m)

    group_starts = {1000: 1, 2000: 5, 4000: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    row_info = [
        ('coverage', 'Two-sided Coverage', 'Actual Coverage'),
        ('width', 'Two-sided CI Width', 'Mean CI Width'),
    ]

    C_pairs = [(0.005, 0.005)]

    K = 200
    for C_a, C_i in C_pairs:
        all_data = {}
        kappa_info = {}
        for case_name, _ in case_info:
            all_data[case_name] = {}
            kappa_info[case_name] = {}
            for n_val in n_values:
                fname = f'alpha_sweep_{case_name}_n{n_val}_Ca{C_a}_Ci{C_i}_v13.npz'
                path = os.path.join(results_dir, fname)
                cov, wid, ka, ki = load_and_compute(path, n_val)
                all_data[case_name][n_val] = (cov, wid)
                kappa_info[case_name][n_val] = (ka, ki)

        n_cols = len(case_info)
        fig, axes = plt.subplots(2, n_cols, figsize=(8, 9))
        if n_cols == 1:
            axes = axes.reshape(2, 1)
        for col, (case_name, case_label) in enumerate(case_info):
            for row, (metric, row_title, ylabel) in enumerate(row_info):
                ax = axes[row, col]
                all_y_lo, all_y_hi = [], []
                for mi in range(n_m):
                    x_vals, y_means, y_stds = [], [], []
                    for n_val in n_values:
                        two_sided_cov, width_pt = all_data[case_name][n_val]
                        pt = two_sided_cov if metric == 'coverage' else width_pt
                        for ci in range(n_conf):
                            xp = x_positions[(n_val, ci)] + jitter[mi]
                            ym = pt[mi, ci, :].mean()
                            ys = pt[mi, ci, :].std()
                            x_vals.append(xp)
                            y_means.append(ym)
                            y_stds.append(ys)
                            all_y_lo.append(ym - ys)
                            all_y_hi.append(ym + ys)
                    if col == 0 and row == 0:
                        label = method_names[mi]
                    else:
                        label = None
                    ax.errorbar(x_vals, y_means, yerr=y_stds,
                                color=colors[mi], marker=markers[mi], markersize=6,
                                linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                                label=label, alpha=0.9)
                if metric == 'coverage':
                    for n_val in n_values:
                        for ci in range(n_conf):
                            xp = x_positions[(n_val, ci)]
                            ax.scatter([xp], [conf_levels[ci]], marker='x',
                                       color='black', s=60, zorder=10, linewidths=1.5)
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
                    J3 = min(int(5 * n_val ** 0.3), K)
                    J4 = min(int(5 * n_val ** 0.4), K)
                    J2 = min(int(5 * n_val ** 0.2), K)
                    ax.text(center, -0.16,
                            f'$n={n_val}$\n$J_{{0.3}}={J3}, J_{{0.4}}={J4}, J_{{0.2}}={J2}$',
                            transform=ax.get_xaxis_transform(),
                            ha='center', fontsize=7, fontweight='bold')
                for sep_x in [4, 8]:
                    ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{row_title} — {case_label}')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
                   bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(rf'$r_2=3$, sparse $\beta=(4,0,-2,0,1,0,\ldots)$ (C1 satisfied): '
                     rf'Trunc $5n^q$ + Adapt.Trunc $5n^{{0.2}}$ (CV, $\mu_n/(\log n)^2$)',
                     fontsize=10, y=1.0)
        plt.tight_layout(rect=[0, 0.08, 1, 0.97])
        out_name = f'inference_alpha_sweep_v13_r2_3_sparse_6m_Ca{C_a}_Ci{C_i}.pdf'
        plt.savefig(os.path.join(results_dir, out_name),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved {out_name}')
