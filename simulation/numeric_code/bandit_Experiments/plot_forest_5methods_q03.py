"""Forest figure for the 5-method (q=0.3-aligned) CIs on 100 GP test functions.

Two-panel main figure:
  A: side-by-side 95% CIs for 15 representative test curves (quantile-spaced
     by Aniso yhat) -- shows method-level overlap and shift.
  B: per-test-function 95% CI half-width with one grey line per test curve --
     shows width comparison across methods.

The legend (5 methods) is emitted as a separate slim landscape PDF, matching
the legend strips in Figures 1-4 of the manuscript. The main figure carries
only panel-specific annotation legends (IQR / highlighted test curves).

Style (rcParams, colours, markers) matches plot_r2_1_grid_inputtest.py for
visual consistency across all simulation/real-data figures.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/Users/zhiyuantang/Dropbox/FunctionalBandit/simulation/numeric_code/bandit_Experiments")
IN = BASE / "std_xnorm_q20_80_logY_ci_5methods_q03_gp_test.csv"
OUT = BASE / "forest_5methods_q03_gp_test_95.pdf"
OUT_LEG = BASE / "forest_5methods_q03_legend.pdf"

CONF_FOREST = 0.95
N_ZOOM = 15

# Match the simulation figures' style (plot_r2_1_grid_inputtest.py).
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

MARKERSIZE = 11        # >5 in the old version
ELINEWIDTH = 1.7
CAPSIZE = 3.5

methods = [
    "Aniso (Lepski)",
    "Iso (Lepski)",
    "FPCA J=sqrt(n)",
    "Trunc.Aniso J=n^0.3 (Lepski)",
    "Adapt.Trunc J=n^0.3",
]
# Display labels matching the manuscript's simulation-figure legend.
display_names = {
    "Aniso (Lepski)":               r"Aniso",
    "Iso (Lepski)":                 r"Iso",
    "FPCA J=sqrt(n)":               r"FPCA",
    "Trunc.Aniso J=n^0.3 (Lepski)": r"Trunc.Aniso",
    "Adapt.Trunc J=n^0.3":          r"Adapt.Trunc",
}
short = {
    "Aniso (Lepski)": "Aniso",
    "Iso (Lepski)": "Iso",
    "FPCA J=sqrt(n)": "FPCA",
    "Trunc.Aniso J=n^0.3 (Lepski)": "Trunc",
    "Adapt.Trunc J=n^0.3": "Adapt",
}
# tab:* palette + marker set borrowed from plot_r2_1_grid_inputtest.py.
palette = {
    "Aniso (Lepski)":                "tab:orange",
    "Iso (Lepski)":                  "tab:green",
    "FPCA J=sqrt(n)":                "tab:purple",
    "Trunc.Aniso J=n^0.3 (Lepski)":  "tab:red",
    "Adapt.Trunc J=n^0.3":           "tab:cyan",
}
markerspec = {
    "Aniso (Lepski)":                "s",
    "Iso (Lepski)":                  "D",
    "FPCA J=sqrt(n)":                "v",
    "Trunc.Aniso J=n^0.3 (Lepski)":  "o",
    "Adapt.Trunc J=n^0.3":           "P",
}


def render_legend_only(out_path):
    """Slim landscape PDF carrying the 5-method legend, matching Figures 1-4."""
    fig = plt.figure(figsize=(14, 0.6))
    proxies = []
    labels = []
    for m in methods:
        h = plt.errorbar([np.nan], [np.nan], yerr=[np.nan],
                         color=palette[m], marker=markerspec[m],
                         markersize=MARKERSIZE,
                         linewidth=0, elinewidth=ELINEWIDTH,
                         capsize=CAPSIZE, capthick=1.1,
                         label=display_names[m])
        proxies.append(h); labels.append(display_names[m])
    plt.axis('off')
    fig.legend(proxies, labels, loc='center', ncol=len(methods),
               bbox_to_anchor=(0.5, 0.5),
               frameon=True, fancybox=False, edgecolor='0.5')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


df = pd.read_csv(IN)

# ----- Panel A: 15-test-function side-by-side forest -----
d95 = df[df.conf_level == CONF_FOREST].copy()
aniso = d95[d95.method == "Aniso (Lepski)"].set_index("test_idx")
order = aniso.yhat.sort_values().index.to_numpy()
zoom_ranks = np.linspace(2, len(order) - 3, N_ZOOM).round().astype(int)
zoom_ids = order[zoom_ranks]
local_rank = {tid: j for j, tid in enumerate(zoom_ids)}
offsets = np.linspace(-0.32, 0.32, len(methods))

fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.38)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

for i, m in enumerate(methods):
    sub = d95[(d95.method == m) & (d95.test_idx.isin(zoom_ids))].copy()
    sub["xr"] = sub["test_idx"].map(local_rank)
    sub = sub.sort_values("xr")
    x = sub["xr"].to_numpy() + offsets[i]
    y = sub.yhat.to_numpy()
    hw = sub.half_width.to_numpy()
    ax_a.errorbar(x, y, yerr=hw,
                  fmt=markerspec[m], ms=MARKERSIZE,
                  linewidth=0, elinewidth=ELINEWIDTH,
                  capsize=CAPSIZE, capthick=1.1,
                  color=palette[m], alpha=0.92)

for j in range(len(zoom_ids)):
    ax_a.axvline(j, color="grey", lw=0.3, alpha=0.4)
ax_a.axhline(0, color="grey", lw=0.6, ls="--")
ax_a.set_xticks(range(len(zoom_ids)))
ax_a.set_xticklabels([f"#{int(t)}" for t in zoom_ids])
ax_a.set_xlabel("test function id (15 evenly-spaced quantiles of Aniso $\\hat y$)")
ax_a.set_ylabel(r"$\hat y \pm 95\%$ CI  (log-kWh)")
ax_a.set_title(r"(a) $95\%$ CIs for $\eta(x)$, 15 representative test curves")
ax_a.grid(alpha=0.2)

# ----- Panel B: half-width distribution by method x confidence level -----
# Boxplot across the 100 GP test curves, grouped per method, three conf
# levels side-by-side. Mirrors the width panels of Figures 1-4.
conf_levels_sorted = sorted(df.conf_level.unique())  # [0.75, 0.85, 0.95]
n_conf = len(conf_levels_sorted)
group_width = 0.78
box_width = group_width / n_conf
hatches = [None, '///', '...']  # B/W-safe distinction across conf levels
conf_alpha = [0.45, 0.65, 0.90]  # darker = larger conf level
for cli, cl in enumerate(conf_levels_sorted):
    d_cl = df[df.conf_level == cl]
    wide_cl = d_cl.pivot(index="test_idx", columns="method",
                         values="half_width")[methods]
    offset = (cli - (n_conf - 1) / 2.0) * box_width
    positions = np.arange(len(methods)) + offset
    box_data = [wide_cl[m].to_numpy() for m in methods]
    bp = ax_b.boxplot(box_data, positions=positions, widths=box_width * 0.85,
                      patch_artist=True, manage_ticks=False,
                      medianprops=dict(color='black', lw=1.6),
                      whiskerprops=dict(lw=1.0),
                      capprops=dict(lw=1.0),
                      flierprops=dict(marker='.', markersize=4,
                                      markerfacecolor='0.4',
                                      markeredgecolor='none', alpha=0.6))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(palette[m])
        patch.set_alpha(conf_alpha[cli])
        patch.set_edgecolor('black')
        patch.set_linewidth(0.9)

ax_b.set_xticks(np.arange(len(methods)))
ax_b.set_xticklabels([short[m] for m in methods])
ax_b.set_xlim(-0.5, len(methods) - 0.5)
ax_b.set_ylabel("CI half-width  (log-kWh)")
ax_b.set_title("(b) CI half-width distribution across $100$ test curves, "
               "by method and nominal level")
ax_b.grid(alpha=0.25, axis="y")
# Tiny conf-level legend (the only in-panel legend kept; methods are in the
# top-strip legend already).
conf_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='0.55', alpha=conf_alpha[cli],
                  edgecolor='black', label=f'{int(round(cl * 100))}\\%')
    for cli, cl in enumerate(conf_levels_sorted)
]
ax_b.legend(handles=conf_handles, title="nominal level",
            loc="upper right", framealpha=0.95, fontsize=12,
            title_fontsize=12)

plt.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")

render_legend_only(OUT_LEG)
