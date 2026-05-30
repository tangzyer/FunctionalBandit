"""3x3 grid figures (rows = slope, cols = covariance design) at multiple
covariance-smoothness levels r_2 in {0.5, 1, 2}, with INPUT-distribution test
functions and the 5 aligned methods (q=0.4 shared by Trunc.Aniso and
Adapt.Trunc).

Writes (per r_2; suffix tag from r2_tag()):
  inference_r2_<tag>_grid_coverage_inputtest.pdf
  inference_r2_<tag>_grid_width_inputtest.pdf
plus one shared legend strip:
  inference_grid_legend_inputtest.pdf

Filename retains the historical 'plot_r2_1_grid_inputtest.py' name from when
it was r_2=1-only; behaviour generalised on 2026-05-30. Visual style mirrors
the `*_combined.pdf` family (markersize=10, serif fonts, axes.linewidth=1.3)
so the grids are legible at journal column width.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


# Style: borrowed from plot_beta4_r2_inputtest_combined.py, tuned down a notch
# because each panel here is denser (9 x-positions x 6 methods vs that
# script's 3 x 5).
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


def load_and_compute(case_name, n_val, results_dir,
                     target_conf_levels=(0.75, 0.85, 0.95), q_adapt=0.4):
    """Return (cov, wid) arrays of shape (5, n_conf, n_test).

    Methods (in display order):
      0: Aniso (Lepski kappa)            -- base v13, m=0
      1: Iso   (Lepski kappa)            -- base v13, m=1
      2: FPCA  sqrt(n)                   -- base v13, m=2
      3: Trunc.Aniso n^{q_adapt}         -- base v13, m=3 if q=0.3 else m=4 if q=0.4
      4: Adapt.Trunc n^{q_adapt}         -- adaptq0p{int(q_adapt*10)} aux NPZ, m=0
    """
    base = np.load(os.path.join(
        results_dir,
        f'alpha_sweep_{case_name}_n{n_val}_Ca0.005_Ci0.005_v13_inputtest.npz'),
        allow_pickle=True)
    aux = np.load(os.path.join(
        results_dir,
        f'alpha_sweep_{case_name}_n{n_val}'
        f'_adaptq0p{int(q_adapt * 10)}_v13_inputtest.npz'),
        allow_pickle=True)
    yhat_base = base['all_yhat']  # (6, datasets, test)
    se_base = base['all_se']
    yhat_aux = aux['all_yhat']    # (1, datasets, test)
    se_aux = aux['all_se']
    true_vals = base['true_vals']
    alphas = base['alphas']

    if abs(q_adapt - 0.3) < 1e-9:
        trunc_idx_in_base = 3
    elif abs(q_adapt - 0.4) < 1e-9:
        trunc_idx_in_base = 4
    else:
        raise ValueError(f'Trunc.Aniso at q={q_adapt} not in base v13 NPZ '
                         '(only q=0.3 and q=0.4 are pre-computed).')

    target_alphas = [1 - cl for cl in target_conf_levels]
    indices = [np.argmin(np.abs(alphas - ta)) for ta in target_alphas]
    n_sel = len(indices)
    n_test = yhat_base.shape[-1]

    cov = np.zeros((5, n_sel, n_test))
    wid = np.zeros((5, n_sel, n_test))

    # (source array, method index in that array, t-quantile dof override)
    J_q = min(int(n_val ** q_adapt), K)
    sources = [
        (yhat_base, se_base, 0, None),                  # Aniso
        (yhat_base, se_base, 1, None),                  # Iso
        (yhat_base, se_base, 2, int(np.sqrt(n_val))),   # FPCA
        (yhat_base, se_base, trunc_idx_in_base, None),  # Trunc.Aniso
        (yhat_aux,  se_aux,  0, J_q),                   # Adapt.Trunc
    ]

    for m_out, (yhat_arr, se_arr, idx, J_for_t) in enumerate(sources):
        yhat = yhat_arr[idx]
        se = se_arr[idx]
        for si, ai in enumerate(indices):
            alpha = alphas[ai]
            if J_for_t is not None:
                df = max(n_val - J_for_t, 1)
                q_two = t_dist.ppf(1 - alpha / 2, df)
            else:
                q_two = norm.ppf(1 - alpha / 2)
            upper = yhat + q_two * se
            lower = yhat - q_two * se
            covered = (true_vals[None, :] >= lower) & (true_vals[None, :] <= upper)
            cov[m_out, si, :] = covered.mean(axis=0)
            w = 2 * q_two * se
            wid[m_out, si, :] = w.mean(axis=0)

    return cov, wid


def r2_tag(r2_val):
    """Map r2 numeric value to the suffix used in NPZ / PDF filenames."""
    if abs(r2_val - 0.5) < 1e-9:
        return '0p5'
    if abs(r2_val - 1.0) < 1e-9:
        return '1'
    if abs(r2_val - 2.0) < 1e-9:
        return '2'
    if abs(r2_val - 3.0) < 1e-9:
        return '3'
    raise ValueError(f'No filename tag mapped for r2={r2_val}')


def slope_case(slope_key, design_key, r2_val):
    """Map (slope, design, r2) -> NPZ case name in run_alpha_sweep_v13."""
    if design_key == 'aligned':
        prefix = f'aligned_r2_{r2_tag(r2_val)}'
    elif design_key == 'haar':
        prefix = f'haar_r2_{r2_tag(r2_val)}'
    elif design_key == 'shifted':
        # Shifted is r2-independent (k0=25 fixed).
        prefix = 'shifted'
    else:
        raise ValueError(design_key)
    if slope_key == 'normal':
        return prefix
    if slope_key == 'beta4':
        return f'{prefix}_beta4'
    if slope_key == 'sparse2':
        return f'{prefix}_sparse2'
    raise ValueError(slope_key)


def render_panel(ax, panel_data, *, quantity, conf_levels, x_positions,
                 n_values, group_starts, jitter, colors, markers, method_names,
                 draw_legend_labels, draw_n_labels):
    n_conf = len(conf_levels)
    n_m = len(method_names)
    series_lo, series_hi = [], []
    for mi in range(n_m):
        x_vals, ym, ys = [], [], []
        for n_val in n_values:
            cov, wid = panel_data[n_val]
            arr = cov if quantity == 'cov' else wid
            for ci in range(n_conf):
                xp = x_positions[(n_val, ci)] + jitter[mi]
                m_val, s_val = arr[mi, ci, :].mean(), arr[mi, ci, :].std()
                x_vals.append(xp); ym.append(m_val); ys.append(s_val)
                series_lo.append(max(m_val - s_val, 1e-12) if quantity == 'wid'
                                 else m_val - s_val)
                series_hi.append(m_val + s_val)
        label = method_names[mi] if draw_legend_labels else None
        ax.errorbar(x_vals, ym, yerr=ys,
                    color=colors[mi], marker=markers[mi],
                    markersize=MARKERSIZE,
                    linewidth=0, elinewidth=ELINEWIDTH,
                    capsize=CAPSIZE, capthick=1.1,
                    label=label, alpha=0.92)

    if quantity == 'cov':
        for cl in conf_levels:
            ax.axhline(cl, color='black', linewidth=0.9, linestyle='--',
                       alpha=0.55, zorder=0)
            series_lo.append(cl); series_hi.append(cl)
        y_lo, y_hi = min(series_lo), max(series_hi)
        pad = (y_hi - y_lo) * 0.04 + 0.005  # tight pad; small floor
        ax.set_ylim(max(0, y_lo - pad), min(1.01, y_hi + pad))
    else:
        y_lo, y_hi = min(series_lo), max(series_hi)
        pad = (y_hi - y_lo) * 0.06
        ax.set_ylim(max(0.0, y_lo - pad), y_hi + pad)

    tick_positions = [x_positions[(n_val, ci)]
                      for n_val in n_values for ci in range(n_conf)]
    tick_labels_list = [f'{conf_levels[ci]:.0%}'
                        for _ in n_values for ci in range(n_conf)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list)
    if draw_n_labels:
        for n_val in n_values:
            center = group_starts[n_val] + 1
            ax.text(center, -0.28, f'$n={n_val}$',
                    transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=14, fontweight='bold')
    for sep_x in [4, 8]:
        ax.axvline(sep_x, color='gray', linewidth=0.6, linestyle=':', alpha=0.55)


def render_legend_only(method_names, colors, markers, out_path):
    """Write a slim landscape PDF containing just the shared legend."""
    fig = plt.figure(figsize=(20, 0.6))
    proxies = []
    for mi, name in enumerate(method_names):
        # Off-canvas errorbar produces a legend handle that matches the
        # in-panel markers exactly (marker, color, error bars, no line).
        h = plt.errorbar([np.nan], [np.nan], yerr=[np.nan],
                         color=colors[mi], marker=markers[mi],
                         markersize=MARKERSIZE,
                         linewidth=0, elinewidth=ELINEWIDTH,
                         capsize=CAPSIZE, capthick=1.1,
                         label=name)
        proxies.append(h)
    plt.axis('off')
    fig.legend(proxies, method_names, loc='center', ncol=len(method_names),
               bbox_to_anchor=(0.5, 0.5),
               frameon=True, fancybox=False, edgecolor='0.5')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.basename(out_path)}')


def render_grid_figure(all_panels, *, quantity, slope_info, design_info,
                       conf_levels, x_positions, n_values, group_starts,
                       jitter, colors, markers, method_names, out_path,
                       suptitle, ylabel, draw_legend=True):
    # Big figure: each cell roughly 6.7 x 5.3 inches like the _combined family
    fig, axes = plt.subplots(3, 3, figsize=(20, 11),
                             sharex='col', sharey='row')
    legend_drawn = False
    n_rows = len(slope_info)
    for ri, (slope_key, slope_label) in enumerate(slope_info):
        for ci, (design_key, design_label) in enumerate(design_info):
            ax = axes[ri, ci]
            panel_data = all_panels[(slope_key, design_key)]
            is_bottom = (ri == n_rows - 1)
            render_panel(ax, panel_data, quantity=quantity,
                         conf_levels=conf_levels, x_positions=x_positions,
                         n_values=n_values, group_starts=group_starts,
                         jitter=jitter, colors=colors, markers=markers,
                         method_names=method_names,
                         draw_legend_labels=(not legend_drawn),
                         draw_n_labels=is_bottom)
            if not legend_drawn:
                legend_drawn = True
            if ri == 0:
                ax.set_title(design_label, pad=8)
            if ci == 0:
                ax.set_ylabel(ylabel)
                # Horizontal row title to the left of the y-axis, multi-line.
                ax.text(-0.27, 0.5, slope_label,
                        transform=ax.transAxes,
                        ha='right', va='center', rotation=0,
                        fontsize=17, fontweight='bold', linespacing=1.3)
    if draw_legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(method_names),
                   bbox_to_anchor=(0.5, -0.01),
                   frameon=True, fancybox=False, edgecolor='0.5')
    # No suptitle: LaTeX \caption{} already supplies it.
    bottom_rect = 0.07 if draw_legend else 0.04
    fig.tight_layout(rect=[0.07, bottom_rect, 1, 0.99])
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.basename(out_path)}')


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    slope_info = [
        ('normal',  r'$b_k\propto k^{-2}$' + '\n(D1 fails)'),
        ('beta4',   r'$b_k\propto k^{-4}$' + '\n(D1 holds)'),
        ('sparse2', r'$(b_1,b_2)=(4,-2)$' + '\n(D1 trivial)'),
    ]
    n_values = [1000, 2000, 4000]
    conf_levels = [0.75, 0.85, 0.95]
    n_conf = len(conf_levels)

    Q_ADAPT = 0.4   # shared truncation level for Trunc.Aniso and Adapt.Trunc
    method_names = [
        r'Aniso',
        r'Iso',
        r'FPCA',
        r'Trunc.Aniso',
        r'Adapt.Trunc',
    ]
    colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:cyan']
    markers = ['s', 'D', 'v', 'o', 'P']
    n_m = len(method_names)
    jitter = np.linspace(-0.30, 0.30, n_m)

    group_starts = {1000: 1, 2000: 5, 4000: 9}
    x_positions = {}
    for n_val in n_values:
        for ci in range(n_conf):
            x_positions[(n_val, ci)] = group_starts[n_val] + ci

    # Legend is the same across all r2 values (same method spec); emit once
    # with a neutral filename and let every r2-figure subfigure include it.
    render_legend_only(
        method_names, colors, markers,
        out_path=os.path.join(
            results_dir, 'inference_grid_legend_inputtest.pdf'),
    )

    R2_VALUES = [0.5, 1.0, 2.0]
    for r2_val in R2_VALUES:
        tag = r2_tag(r2_val)
        # Pretty r2 for the design column titles.
        r2_pretty = (f'{r2_val:.1f}'.rstrip('0').rstrip('.')
                     if r2_val != int(r2_val) else f'{int(r2_val)}')
        design_info = [
            ('aligned',  rf'Aligned ($r_2={r2_pretty}$)'),
            ('haar',     rf'Haar ($r_2={r2_pretty}$)'),
            ('shifted',  r'Shifted ($k_0=25$)'),
        ]

        all_panels = {}
        for slope_key, _ in slope_info:
            for design_key, _ in design_info:
                case = slope_case(slope_key, design_key, r2_val)
                panel = {}
                for n_val in n_values:
                    cov, wid = load_and_compute(
                        case, n_val, results_dir, q_adapt=Q_ADAPT)
                    panel[n_val] = (cov, wid)
                all_panels[(slope_key, design_key)] = panel

        render_grid_figure(
            all_panels, quantity='cov',
            slope_info=slope_info, design_info=design_info,
            conf_levels=conf_levels, x_positions=x_positions, n_values=n_values,
            group_starts=group_starts, jitter=jitter, colors=colors,
            markers=markers, method_names=method_names,
            out_path=os.path.join(
                results_dir, f'inference_r2_{tag}_grid_coverage_inputtest.pdf'),
            suptitle=(rf'Two-sided coverage at $r_2={r2_pretty}$ '
                      r'(input-distribution test points)'),
            ylabel='Actual coverage',
            draw_legend=False,  # Legend lives in the separate legend PDF.
        )
        render_grid_figure(
            all_panels, quantity='wid',
            slope_info=slope_info, design_info=design_info,
            conf_levels=conf_levels, x_positions=x_positions, n_values=n_values,
            group_starts=group_starts, jitter=jitter, colors=colors,
            markers=markers, method_names=method_names,
            out_path=os.path.join(
                results_dir, f'inference_r2_{tag}_grid_width_inputtest.pdf'),
            suptitle=(rf'Mean CI width at $r_2={r2_pretty}$ '
                      r'(input-distribution test points)'),
            ylabel='Mean CI width',
            draw_legend=False,
        )
