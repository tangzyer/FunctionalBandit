"""Plot alpha sweep results with error bars, tight y-axis, and broken axis for gaps."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, t as t_dist


def load_and_compute(npz_path):
    """Load saved (y_hat, se) data and compute per-test-function coverage/width."""
    data = np.load(npz_path, allow_pickle=True)
    all_yhat = data['all_yhat']    # (n_methods, n_datasets, n_test)
    all_se = data['all_se']        # (n_methods, n_datasets, n_test)
    true_vals = data['true_vals']  # (n_test,)
    alphas = data['alphas']        # (30,)

    n_methods, n_datasets, n_test = all_yhat.shape
    n_alpha = len(alphas)

    cov_per_test = np.zeros((n_methods, n_alpha, n_test))
    width_per_test = np.zeros((n_methods, n_alpha, n_test))

    for m in range(n_methods):
        for ai, alpha in enumerate(alphas):
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
            cov_per_test[m, ai, :] = covered.mean(axis=0)
            width_per_test[m, ai, :] = width.mean(axis=0)

    return alphas, cov_per_test, width_per_test


def _find_gap(means, stds):
    """Check if data has a large gap. Returns (has_gap, low_range, high_range)."""
    n_methods = means.shape[0]
    # Collect all (mean-std, mean+std) ranges per method
    ranges = []
    for m in range(n_methods):
        lo = (means[m] - stds[m]).min()
        hi = (means[m] + stds[m]).max()
        ranges.append((lo, hi))

    # Sort by lower bound
    ranges.sort(key=lambda x: x[0])

    # Check for gap between consecutive range clusters
    for i in range(len(ranges) - 1):
        gap_bottom = ranges[i][1]
        gap_top = ranges[i + 1][0]
        total_span = ranges[-1][1] - ranges[0][0]
        if total_span > 0 and (gap_top - gap_bottom) / total_span > 0.25:
            # Found significant gap
            low_top = gap_bottom + 0.03
            high_bottom = gap_top - 0.03
            return True, (max(0, ranges[0][0] - 0.02), low_top), (high_bottom, min(1.05, ranges[-1][1] + 0.02))

    return False, None, None


def _plot_panel_broken(fig, gs_slot, conf_levels, means, stds, colors, markers,
                       method_names, title, ylabel, low_range, high_range,
                       show_legend=False, show_ideal=False):
    """Plot a panel with broken y-axis."""
    # Height ratio: proportional to range size
    low_span = low_range[1] - low_range[0]
    high_span = high_range[1] - high_range[0]
    ratio = high_span / (low_span + high_span)

    inner_gs = gs_slot.subgridspec(2, 1, height_ratios=[ratio, 1 - ratio], hspace=0.08)
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bot = fig.add_subplot(inner_gs[1])

    for ax, ylim in [(ax_top, high_range), (ax_bot, low_range)]:
        if show_ideal:
            ax.plot(conf_levels, conf_levels, 'k--', alpha=0.4, linewidth=1,
                    label='Ideal' if ax is ax_top else None)
        for m in range(means.shape[0]):
            ax.errorbar(conf_levels, means[m], yerr=stds[m],
                        color=colors[m], marker=markers[m], markersize=4,
                        linewidth=1.2, capsize=2, capthick=0.8,
                        label=method_names[m] if ax is ax_top else None,
                        alpha=0.85)
        ax.set_ylim(ylim)

    # Hide spines and add break marks
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ax_bot.set_xlabel(r'Confidence Level $(1-\alpha)$')

    # Break marks
    d = 0.01
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.set_title(title)
    ax_top.set_ylabel(ylabel)
    if show_legend:
        ax_top.legend(fontsize=7, loc='upper left')

    return ax_top, ax_bot


def _plot_panel_normal(fig, gs_slot, conf_levels, means, stds, colors, markers,
                       method_names, title, ylabel, show_legend=False, show_ideal=False):
    """Plot a normal panel (no broken axis)."""
    ax = fig.add_subplot(gs_slot)

    if show_ideal:
        ax.plot(conf_levels, conf_levels, 'k--', alpha=0.4, label='Ideal', linewidth=1)

    y_lo_all, y_hi_all = [], []
    for m in range(means.shape[0]):
        ax.errorbar(conf_levels, means[m], yerr=stds[m],
                    color=colors[m], marker=markers[m], markersize=4,
                    linewidth=1.2, capsize=2, capthick=0.8,
                    label=method_names[m], alpha=0.85)
        y_lo_all.append((means[m] - stds[m]).min())
        y_hi_all.append((means[m] + stds[m]).max())

    pad = (max(y_hi_all) - min(y_lo_all)) * 0.06
    ax.set_ylim(max(0, min(y_lo_all) - pad), max(y_hi_all) + pad)
    ax.set_xlabel(r'Confidence Level $(1-\alpha)$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=7, loc='upper left')
    return ax


def plot_alpha_sweep(case_data, case_labels, save_path):
    method_names = [
        r'Aniso oracle $\lambda$',
        r'Aniso $\lambda/(\log n)^2$',
        r'Iso oracle $\lambda$',
        r'Iso $\lambda/(\log n)^2$',
        r'FPCA $\sqrt{n}$',
    ]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']

    n_cases = len(case_data)
    fig = plt.figure(figsize=(6 * n_cases, 10))
    gs = GridSpec(2, n_cases, figure=fig, hspace=0.35, wspace=0.3)

    for col, (alphas, cov_pt, width_pt) in enumerate(case_data):
        conf_levels = 1 - alphas
        n_methods = cov_pt.shape[0]

        cov_mean = cov_pt.mean(axis=2)
        cov_std = cov_pt.std(axis=2)
        width_mean = width_pt.mean(axis=2)
        width_std = width_pt.std(axis=2)

        # Coverage panel
        has_gap, low_r, high_r = _find_gap(cov_mean, cov_std)
        if has_gap:
            _plot_panel_broken(fig, gs[0, col], conf_levels, cov_mean, cov_std,
                               colors, markers, method_names,
                               f'Coverage — {case_labels[col]}', 'Actual Coverage',
                               low_r, high_r, show_legend=(col == 0), show_ideal=True)
        else:
            _plot_panel_normal(fig, gs[0, col], conf_levels, cov_mean, cov_std,
                               colors, markers, method_names,
                               f'Coverage — {case_labels[col]}', 'Actual Coverage',
                               show_legend=(col == 0), show_ideal=True)

        # Width panel
        has_gap_w, low_rw, high_rw = _find_gap(width_mean, width_std)
        if has_gap_w:
            _plot_panel_broken(fig, gs[1, col], conf_levels, width_mean, width_std,
                               colors, markers, method_names,
                               f'CI Width — {case_labels[col]}', 'Mean CI Width',
                               low_rw, high_rw, show_legend=(col == 0))
        else:
            _plot_panel_normal(fig, gs[1, col], conf_levels, width_mean, width_std,
                               colors, markers, method_names,
                               f'CI Width — {case_labels[col]}', 'Mean CI Width',
                               show_legend=(col == 0))

    fig.suptitle(r'Coverage & CI Width vs Confidence Level $(1-\alpha)$, $n = 256$',
                 fontsize=14, y=0.98)
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
        save_path = os.path.join(results_dir, 'inference_alpha_sweep_v2.pdf')
        plot_alpha_sweep(cases, labels, save_path)
    else:
        print('No alpha sweep data found!')
