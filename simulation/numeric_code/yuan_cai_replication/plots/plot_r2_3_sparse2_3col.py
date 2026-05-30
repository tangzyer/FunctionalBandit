"""Plot coverage/width for sparse beta=(4,-2,0,...) (C1 satisfied), 6 methods, J=n^q.
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

    two_sided_cov = np.zeros((n_methods, n_sel, n_test))
    width_per_test = np.zeros((n_methods, n_sel, n_test))

    K = 200
    for m in range(n_methods):
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if m == 2:
                J = int(np.sqrt(n_val))
                df = n_val - J
                q_two = t_dist.ppf(1 - alpha / 2, df)
            elif m == 5:
                J = min(int(n_val ** 0.2), K)
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
    K = 200

    case_info = [
        ('aligned_r2_3_sparse2', r'Aligned ($r_2=3$)'),
        ('haar_r2_3_sparse2', r'Haar ($r_2=3$)'),
        ('shifted_sparse2', r'Shifted ($k_0=25$)'),
    ]
    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    method_names = [
        r'Aniso (Lepski $\kappa$)',
        r'Iso (Lepski $\kappa$)',
        r'FPCA $\sqrt{n}$',
        r'Trunc.Aniso $n^{0.3}$ (Lepski $\kappa$)',
        r'Trunc.Aniso $n^{0.4}$ (Lepski $\kappa$)',
        r'Adapt.Trunc $n^{0.2}$',
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

    C_a, C_i = 0.005, 0.005
    all_data = {}
    for case_name, _ in case_info:
        all_data[case_name] = {}
        for n_val in n_values:
            fname = f'alpha_sweep_{case_name}_n{n_val}_Ca{C_a}_Ci{C_i}_v13.npz'
            path = os.path.join(results_dir, fname)
            cov, wid = load_and_compute(path, n_val)
            all_data[case_name][n_val] = (cov, wid)

    from matplotlib.gridspec import GridSpec
    n_cols = len(case_info)
    fig = plt.figure(figsize=(18, 10))
    outer = GridSpec(2, n_cols, figure=fig,
                     height_ratios=[3.0, 4.0], hspace=0.28)
    # Per-column: split only if FPCA width >> others (ratio > 3 at largest n)
    needs_split = []
    for case_name, _ in case_info:
        fpca_w = max(all_data[case_name][n][1][2].mean() for n in n_values)
        other_w = max(all_data[case_name][n][1][m].mean()
                      for n in n_values for m in [0, 1, 3, 4, 5])
        needs_split.append(fpca_w > 10 * other_w)
    axes_cov = [fig.add_subplot(outer[0, c]) for c in range(n_cols)]
    axes_wt, axes_wb = [], []
    for c in range(n_cols):
        if needs_split[c]:
            inner = outer[1, c].subgridspec(2, 1, hspace=0.06,
                                            height_ratios=[1.2, 3.0])
            axes_wt.append(fig.add_subplot(inner[0]))
            axes_wb.append(fig.add_subplot(inner[1]))
        else:
            axes_wt.append(None)
            axes_wb.append(fig.add_subplot(outer[1, c]))
    axes_cov = np.array([axes_cov])

    for col, (case_name, case_label) in enumerate(case_info):
        ax_cov = axes_cov[0, col]; ax_wt = axes_wt[col]; ax_wb = axes_wb[col]
        split_here = needs_split[col]
        cov_lo, cov_hi = [], []
        wt_lo, wt_hi = [], []   # FPCA width bounds (top axis)
        wb_lo, wb_hi = [], []   # non-FPCA width bounds (bottom axis)

        for mi in range(n_m):
            x_vals = []; ym_c = []; ys_c = []; ym_w = []; ys_w = []
            for n_val in n_values:
                cov, wid = all_data[case_name][n_val]
                for ci in range(n_conf):
                    xp = x_positions[(n_val, ci)] + jitter[mi]
                    mc, sc = cov[mi, ci, :].mean(), cov[mi, ci, :].std()
                    mw, sw = wid[mi, ci, :].mean(), wid[mi, ci, :].std()
                    x_vals.append(xp)
                    ym_c.append(mc); ys_c.append(sc)
                    ym_w.append(mw); ys_w.append(sw)
                    cov_lo.append(mc - sc); cov_hi.append(mc + sc)
                    if mi == 2:
                        wt_lo.append(mw - sw); wt_hi.append(mw + sw)
                    else:
                        wb_lo.append(mw - sw); wb_hi.append(mw + sw)
            label = method_names[mi] if col == 0 else None
            ax_cov.errorbar(x_vals, ym_c, yerr=ys_c,
                            color=colors[mi], marker=markers[mi], markersize=6,
                            linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                            label=label, alpha=0.9)
            # When split: FPCA only on upper axis, all others only on lower.
            # When not split: all methods on the single (lower) axis.
            if split_here:
                target_ax = ax_wt if mi == 2 else ax_wb
            else:
                target_ax = ax_wb
            target_ax.errorbar(x_vals, ym_w, yerr=ys_w,
                               color=colors[mi], marker=markers[mi], markersize=6,
                               linewidth=0, elinewidth=1.3, capsize=3, capthick=1,
                               alpha=0.9)

        for cl in conf_levels:
            ax_cov.axhline(cl, color='black', linewidth=0.8, linestyle='--',
                           alpha=0.5, zorder=0)
            cov_lo.append(cl); cov_hi.append(cl)

        # Coverage y-lim
        y_lo, y_hi = min(cov_lo), max(cov_hi); pad = (y_hi - y_lo) * 0.06
        ax_cov.set_ylim(max(0, y_lo - pad), min(1.02, y_hi + pad))
        # Width top (FPCA) only if splitting
        if split_here and wt_lo:
            y_lo, y_hi = min(wt_lo), max(wt_hi)
            pad = (y_hi - y_lo) * 0.1 if y_hi > y_lo else max(1.0, 0.05 * y_hi)
            ax_wt.set_ylim(max(0, y_lo - pad), y_hi + pad)
        # Width bottom: non-FPCA when split; all methods otherwise
        if split_here:
            bounds_lo, bounds_hi = wb_lo, wb_hi
        else:
            bounds_lo = wb_lo + wt_lo
            bounds_hi = wb_hi + wt_hi
        y_lo, y_hi = min(bounds_lo), max(bounds_hi)
        pad = (y_hi - y_lo) * 0.1 if y_hi > y_lo else max(0.05, 0.1 * y_hi)
        ax_wb.set_ylim(max(0, y_lo - pad), y_hi + pad)

        # Break-axis styling only when split
        if split_here:
            ax_wt.spines['bottom'].set_visible(False)
            ax_wb.spines['top'].set_visible(False)
            ax_wt.tick_params(bottom=False, labelbottom=False)
            d = 0.012
            kwargs = dict(transform=ax_wt.transAxes, color='k',
                          clip_on=False, lw=0.8)
            ax_wt.plot((-d, +d), (-4*d, +4*d), **kwargs)
            ax_wt.plot((1-d, 1+d), (-4*d, +4*d), **kwargs)
            kwargs = dict(transform=ax_wb.transAxes, color='k',
                          clip_on=False, lw=0.8)
            ax_wb.plot((-d, +d), (1-4*d, 1+4*d), **kwargs)
            ax_wb.plot((1-d, 1+d), (1-4*d, 1+4*d), **kwargs)

        tick_positions = [x_positions[(n_val, ci)]
                          for n_val in n_values for ci in range(n_conf)]
        tick_labels_list = [f'{conf_levels[ci]:.0%}'
                            for _ in n_values for ci in range(n_conf)]
        for ax in (ax_cov, ax_wb):
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
        if split_here:
            ax_wt.set_xticks(tick_positions)
            ax_wt.set_xlim(ax_wb.get_xlim())
            for sep_x in [4, 8]:
                ax_wt.axvline(sep_x, color='gray', linewidth=0.5,
                              linestyle=':', alpha=0.5)

        ax_cov.set_ylabel('Actual Coverage')
        ax_wb.set_ylabel('Mean CI Width')
        ax_cov.set_title(f'Two-sided Coverage — {case_label}')
        if split_here:
            ax_wt.set_ylabel('FPCA', fontsize=9)
            ax_wt.set_title(f'Two-sided CI Width — {case_label}', fontsize=10)
        else:
            ax_wb.set_title(f'Two-sided CI Width — {case_label}', fontsize=10)

    handles, labels = axes_cov[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(rf'Sparse $\beta=(4,-2,0,\ldots)$ (C1 satisfied), $J=n^q$',
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out_name = 'inference_sparse2_3col_6m.pdf'
    plt.savefig(os.path.join(results_dir, out_name),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {out_name}')
