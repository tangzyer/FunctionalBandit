"""Truncation-level sensitivity figure (beta4 slope, INPUT-distribution test
functions). 3x3 grid: rows = r_2 in {0.5, 1, 2}, cols = covariance design
(Aligned, Haar, Shifted). Within each panel, six series: Trunc.Aniso and
Adapt.Trunc, each at q in {0.2, 0.3, 0.4}.

Method encoded by marker shape, q encoded by colour shade (light -> dark for
0.2 -> 0.3 -> 0.4) so the ordering across q is visually intuitive.

Writes three PDFs (matching the convention of the main grid figure):
  inference_trunc_sensitivity_legend_inputtest.pdf
  inference_trunc_sensitivity_coverage_inputtest.pdf
  inference_trunc_sensitivity_width_inputtest.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


# Style mirrors plot_r2_1_grid_inputtest.py.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 12,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
    'axes.linewidth': 1.3,
    'lines.markeredgewidth': 0.9,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
})

MARKERSIZE = 10
ELINEWIDTH = 1.5
CAPSIZE = 3.2
K = 200

# Method by marker shape, q by saturated print-safe colour.
# Same colour for the same q across the two methods, so the reader can
# pair (square, circle) at a glance. Avoiding light/dark shading because
# faint shades blur on monochrome printers.
COLOR_BY_Q = {0.2: 'tab:blue', 0.3: 'tab:green', 0.4: 'tab:red'}
MARKER_TRUNC = 's'
MARKER_ADAPT = 'o'


def _q_two(alpha, J_for_t, n_val):
    if J_for_t is not None:
        df = max(n_val - J_for_t, 1)
        return t_dist.ppf(1 - alpha / 2, df)
    return norm.ppf(1 - alpha / 2)


def load_panel(case_name, n_val, results_dir,
               q_values=(0.2, 0.3, 0.4),
               target_conf_levels=(0.75, 0.85, 0.95)):
    """Return cov, wid arrays of shape (6, n_conf, n_test).

    Series order (matches LEGEND_ENTRIES below):
      0: Trunc.Aniso q=0.2  -- truncq0p20 NPZ, m=0
      1: Trunc.Aniso q=0.3  -- base v13 NPZ, m=3
      2: Trunc.Aniso q=0.4  -- base v13 NPZ, m=4
      3: Adapt.Trunc q=0.2  -- adaptq0p2 NPZ, m=0
      4: Adapt.Trunc q=0.3  -- adaptq0p3 NPZ, m=0
      5: Adapt.Trunc q=0.4  -- adaptq0p4 NPZ, m=0
    """
    base = np.load(os.path.join(
        results_dir,
        f'alpha_sweep_{case_name}_n{n_val}_Ca0.005_Ci0.005_v13_inputtest.npz'),
        allow_pickle=True)
    yhat_base = base['all_yhat']
    se_base = base['all_se']
    true_vals = base['true_vals']
    alphas = base['alphas']

    aux_arrays = {}
    for q in q_values:
        if abs(q - 0.2) < 1e-9:
            # Trunc.Aniso at q=0.2 lives in truncq0p20_v13_inputtest.npz.
            tr_path = os.path.join(
                results_dir,
                f'alpha_sweep_{case_name}_n{n_val}'
                f'_truncq0p20_v13_inputtest.npz')
            tr = np.load(tr_path, allow_pickle=True)
            aux_arrays[('trunc', q)] = (tr['all_yhat'][0], tr['all_se'][0])
        # Adapt.Trunc always comes from adaptq0pX file.
        ad_path = os.path.join(
            results_dir,
            f'alpha_sweep_{case_name}_n{n_val}'
            f'_adaptq0p{int(q * 10)}_v13_inputtest.npz')
        ad = np.load(ad_path, allow_pickle=True)
        aux_arrays[('adapt', q)] = (ad['all_yhat'][0], ad['all_se'][0])

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)
    n_test = yhat_base.shape[-1]

    cov = np.zeros((6, n_sel, n_test))
    wid = np.zeros((6, n_sel, n_test))

    # (yhat, se, J_for_t)
    def get_source(method, q):
        J_q = min(int(n_val ** q), K)
        if method == 'trunc':
            if abs(q - 0.2) < 1e-9:
                y, s = aux_arrays[('trunc', q)]
            elif abs(q - 0.3) < 1e-9:
                y, s = yhat_base[3], se_base[3]
            elif abs(q - 0.4) < 1e-9:
                y, s = yhat_base[4], se_base[4]
            else:
                raise ValueError(q)
            return y, s, None       # Trunc.Aniso uses normal quantile
        # adapt
        y, s = aux_arrays[('adapt', q)]
        return y, s, J_q            # Adapt.Trunc uses Student-t (n - J dof)

    series = []
    for q in q_values:
        series.append(('trunc', q))
    for q in q_values:
        series.append(('adapt', q))

    for m_out, (method, q) in enumerate(series):
        yhat, se, J_for_t = get_source(method, q)
        for si, ai in enumerate(indices):
            qt = _q_two(alphas[ai], J_for_t, n_val)
            upper = yhat + qt * se
            lower = yhat - qt * se
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            cov[m_out, si, :] = covered.mean(axis=0)
            wid[m_out, si, :] = (2 * qt * se).mean(axis=0)
    return cov, wid


# Series metadata used by both the rendering and the legend strip.
Q_VALUES = (0.2, 0.3, 0.4)
SERIES = (
    [('trunc', q, MARKER_TRUNC, COLOR_BY_Q[q],
      rf'Trunc.Aniso $J=n^{{{q}}}$') for q in Q_VALUES]
    + [('adapt', q, MARKER_ADAPT, COLOR_BY_Q[q],
        rf'Adapt.Trunc $J=n^{{{q}}}$') for q in Q_VALUES]
)
N_SERIES = len(SERIES)


def render_panel(ax, panel_data, *, quantity, conf_levels, x_positions,
                 n_values, group_starts, jitter, draw_legend_labels,
                 draw_n_labels):
    n_conf = len(conf_levels)
    series_lo, series_hi = [], []
    for mi, (_, _, marker, color, label) in enumerate(SERIES):
        x_vals, ym, ys = [], [], []
        for n_val in n_values:
            cov, wid = panel_data[n_val]
            arr = cov if quantity == 'cov' else wid
            for ci in range(n_conf):
                xp = x_positions[(n_val, ci)] + jitter[mi]
                m_val, s_val = arr[mi, ci, :].mean(), arr[mi, ci, :].std()
                x_vals.append(xp); ym.append(m_val); ys.append(s_val)
                series_lo.append(m_val - s_val)
                series_hi.append(m_val + s_val)
        ax.errorbar(x_vals, ym, yerr=ys,
                    color=color, marker=marker, markersize=MARKERSIZE,
                    linewidth=0, elinewidth=ELINEWIDTH,
                    capsize=CAPSIZE, capthick=1.1,
                    label=label if draw_legend_labels else None,
                    alpha=0.92)

    if quantity == 'cov':
        for cl in conf_levels:
            ax.axhline(cl, color='black', linewidth=0.9, linestyle='--',
                       alpha=0.55, zorder=0)
            series_lo.append(cl); series_hi.append(cl)
        y_lo, y_hi = min(series_lo), max(series_hi)
        pad = (y_hi - y_lo) * 0.04 + 0.005
        ax.set_ylim(max(0, y_lo - pad), min(1.01, y_hi + pad))
    else:
        y_lo, y_hi = min(series_lo), max(series_hi)
        pad = (y_hi - y_lo) * 0.06
        ax.set_ylim(max(0.0, y_lo - pad), y_hi + pad)

    tick_positions = [x_positions[(n_val, ci)]
                      for n_val in n_values for ci in range(n_conf)]
    tick_labels = [f'{conf_levels[ci]:.0%}'
                   for _ in n_values for ci in range(n_conf)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    if draw_n_labels:
        for n_val in n_values:
            center = group_starts[n_val] + 1
            ax.text(center, -0.28, f'$n={n_val}$',
                    transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=14, fontweight='bold')
    for sep_x in [4, 8]:
        ax.axvline(sep_x, color='gray', linewidth=0.6, linestyle=':', alpha=0.55)


def render_grid_figure(all_panels, *, quantity, r2_info, design_info,
                       conf_levels, x_positions, n_values, group_starts,
                       jitter, out_path, ylabel):
    # rows = design, cols = r2 -- so the reader scans horizontally to see how
    # widths change (or don't) with r2 within a fixed design and fixed y-axis.
    n_rows = len(design_info)
    n_cols = len(r2_info)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.7 * n_cols, 4.0 * n_rows),
                             sharex='col', sharey='row')
    legend_drawn = False
    for ri, (design_key, design_label) in enumerate(design_info):
        for ci, (r2_val, r2_label) in enumerate(r2_info):
            ax = axes[ri, ci]
            panel_data = all_panels[(r2_val, design_key)]
            is_bottom = (ri == n_rows - 1)
            render_panel(ax, panel_data, quantity=quantity,
                         conf_levels=conf_levels, x_positions=x_positions,
                         n_values=n_values, group_starts=group_starts,
                         jitter=jitter,
                         draw_legend_labels=(not legend_drawn),
                         draw_n_labels=is_bottom)
            if not legend_drawn:
                legend_drawn = True
            if ri == 0:
                ax.set_title(r2_label, pad=8)
            if ci == 0:
                ax.set_ylabel(ylabel)
                ax.text(-0.27, 0.5, design_label,
                        transform=ax.transAxes,
                        ha='right', va='center', rotation=0,
                        fontsize=17, fontweight='bold', linespacing=1.3)
    fig.tight_layout(rect=[0.07, 0.06, 1, 0.99])
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.basename(out_path)}')


def render_legend_only(out_path):
    # Width matched to the new 2x3 grid figure (6.7 * 3 = 20.1 inches wide).
    fig = plt.figure(figsize=(20.1, 0.6))
    proxies = []
    labels = []
    for _, _, marker, color, label in SERIES:
        h = plt.errorbar([np.nan], [np.nan], yerr=[np.nan],
                         color=color, marker=marker, markersize=MARKERSIZE,
                         linewidth=0, elinewidth=ELINEWIDTH,
                         capsize=CAPSIZE, capthick=1.1, label=label)
        proxies.append(h); labels.append(label)
    plt.axis('off')
    fig.legend(proxies, labels, loc='center', ncol=N_SERIES,
               bbox_to_anchor=(0.5, 0.5),
               frameon=True, fancybox=False, edgecolor='0.5')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.basename(out_path)}')


def _design_case(design_key, r2_val):
    """Map (design, r2) -> beta4-slope case_name."""
    if design_key == 'aligned':
        return f'aligned_r2_{_r2_tag(r2_val)}_beta4'
    if design_key == 'haar':
        return f'haar_r2_{_r2_tag(r2_val)}_beta4'
    if design_key == 'shifted':
        return 'shifted_beta4'
    raise ValueError(design_key)


def _r2_tag(r2_val):
    if abs(r2_val - 0.5) < 1e-9:
        return '0p5'
    if abs(r2_val - 1.0) < 1e-9:
        return '1'
    if abs(r2_val - 2.0) < 1e-9:
        return '2'
    raise ValueError(r2_val)


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    r2_info = [
        (0.5, r'$r_2=0.5$'),
        (1.0, r'$r_2=1$'),
        (2.0, r'$r_2=2$'),
    ]
    design_info = [
        ('aligned',  r'Aligned'),
        ('haar',     r'Haar'),
    ]
    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    jitter = np.linspace(-0.32, 0.32, N_SERIES)
    group_starts = {1000: 1, 2000: 5, 4000: 9}
    x_positions = {(n_val, ci): group_starts[n_val] + ci
                   for n_val in n_values for ci in range(n_conf)}

    all_panels = {}
    for r2_val, _ in r2_info:
        for design_key, _ in design_info:
            case = _design_case(design_key, r2_val)
            panel = {}
            for n_val in n_values:
                cov, wid = load_panel(case, n_val, results_dir)
                panel[n_val] = (cov, wid)
            all_panels[(r2_val, design_key)] = panel

    render_legend_only(
        out_path=os.path.join(
            results_dir, 'inference_trunc_sensitivity_legend_inputtest.pdf'),
    )
    render_grid_figure(
        all_panels, quantity='cov',
        r2_info=r2_info, design_info=design_info,
        conf_levels=conf_levels, x_positions=x_positions, n_values=n_values,
        group_starts=group_starts, jitter=jitter,
        out_path=os.path.join(
            results_dir, 'inference_trunc_sensitivity_coverage_inputtest.pdf'),
        ylabel='Actual coverage',
    )
    render_grid_figure(
        all_panels, quantity='wid',
        r2_info=r2_info, design_info=design_info,
        conf_levels=conf_levels, x_positions=x_positions, n_values=n_values,
        group_starts=group_starts, jitter=jitter,
        out_path=os.path.join(
            results_dir, 'inference_trunc_sensitivity_width_inputtest.pdf'),
        ylabel='Mean CI width',
    )
