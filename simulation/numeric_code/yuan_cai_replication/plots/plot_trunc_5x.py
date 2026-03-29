"""Plot coverage/width for Trunc.Aniso with J=5*n^0.3 and J=5*n^0.4."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist


def load_and_compute(npz_path, n_val, target_conf_levels=(0.75, 0.85, 0.95)):
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']
    all_se = data['all_se']
    true_vals = data['true_vals']
    alphas = data['alphas']
    J_vals = data['J_vals']
    n_methods, n_datasets, n_test = all_yhat.shape

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)

    two_sided_cov = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    for m in range(n_methods):
        J = int(J_vals[m])
        df = n_val - J
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            q_two = t_dist.ppf(1 - alpha / 2, df)

            upper = all_yhat[m] + q_two * all_se[m]
            lower = all_yhat[m] - q_two * all_se[m]
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            two_sided_cov[m, si, :] = covered.mean(axis=0)

            width = 2 * q_two * all_se[m]
            width_per_test[m, si, :] = width.mean(axis=0)

    return two_sided_cov, width_per_test, J_vals


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    case_info = [
        ('aligned_r2_2', r'Aligned ($r_2=2$)'),
        ('shifted', r'Shifted $k_0=50$'),
        ('haar_r2_2', r'Haar ($r_2=2$)'),
    ]
    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    method_names = [
        r'Trunc.Aniso $5 n^{0.3}$',
        r'Trunc.Aniso $5 n^{0.4}$',
    ]
    colors = ['tab:blue', 'tab:red']
    markers = ['^', 'o']
    n_m = len(method_names)
    jitter = np.linspace(-0.12, 0.12, n_m)

    group_starts = {1000: 1, 2000: 5, 4000: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    row_info = [
        ('coverage', 'Two-sided Coverage', 'Actual Coverage'),
        ('width', 'Two-sided CI Width', 'Mean CI Width'),
    ]

    all_data = {}
    j_info = {}
    for case_name, _ in case_info:
        all_data[case_name] = {}
        j_info[case_name] = {}
        for n_val in n_values:
            fname = f'trunc_5x_{case_name}_n{n_val}.npz'
            path = os.path.join(results_dir, fname)
            cov, wid, J_vals = load_and_compute(path, n_val)
            all_data[case_name][n_val] = (cov, wid)
            j_info[case_name][n_val] = J_vals

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
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
                            color=colors[mi], marker=markers[mi], markersize=7,
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
                Jv = j_info[case_name][n_val]
                ax.text(center, -0.16,
                        f'$n={n_val}$\n$J_{{0.3}}={int(Jv[0])}, J_{{0.4}}={int(Jv[1])}$',
                        transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=7, fontweight='bold')
            for sep_x in [4, 8]:
                ax.axvline(sep_x, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{row_title} — {case_label}')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r'Trunc.Aniso with $J = 5 n^q$ ($b_k \sim k^{-2}$, RKHS $r=2$)',
                 fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out_name = 'inference_trunc_5x.pdf'
    plt.savefig(os.path.join(results_dir, out_name),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {out_name}')
