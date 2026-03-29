"""Plot Figure 3: Misalignment scenarios (6-panel figure)."""

import os
import numpy as np
import matplotlib.pyplot as plt


MARKERS = ['o', 's', '^', 'D', 'v']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def _plot_row(axes, n_values, param_values, risk_reg, risk_fpca,
              param_label, param_format):
    """Plot one row of 3 panels: Reg risk, FPCA risk, Relative efficiency."""

    # Panel 1: Roughness Reg
    ax = axes[0]
    for i, pval in enumerate(param_values):
        means = np.mean(risk_reg[i], axis=1)
        se = np.std(risk_reg[i], axis=1) / np.sqrt(risk_reg.shape[2])
        ax.errorbar(
            n_values, means, yerr=se,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=param_format(pval),
            capsize=3, linewidth=1.5, markersize=5,
        )
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Excess Risk')
    ax.set_title('Roughness Regularization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: FPCA
    ax = axes[1]
    for i, pval in enumerate(param_values):
        means = np.mean(risk_fpca[i], axis=1)
        se = np.std(risk_fpca[i], axis=1) / np.sqrt(risk_fpca.shape[2])
        ax.errorbar(
            n_values, means, yerr=se,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=param_format(pval),
            capsize=3, linewidth=1.5, markersize=5,
        )
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Excess Risk')
    ax.set_title('FPCA')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Relative efficiency (median risk_FPCA / median risk_Reg)
    ax = axes[2]
    for i, pval in enumerate(param_values):
        median_reg = np.median(risk_reg[i], axis=1)
        median_fpca = np.median(risk_fpca[i], axis=1)
        ratio = median_fpca / median_reg
        ax.plot(
            n_values, ratio,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=param_format(pval),
            linewidth=1.5, markersize=5,
        )
    ax.set_xscale('log', base=2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('n')
    ax.set_ylabel('Relative Efficiency\n(FPCA / Reg)')
    ax.set_title('Relative Efficiency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_figure3(top_path, bottom_path, save_path=None):
    """Create 6-panel Figure 3 (2 rows x 3 columns)."""
    top = np.load(top_path)
    bot = np.load(bottom_path)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: shifted k₀
    _plot_row(
        axes[0],
        top['n_values'], top['k0_values'],
        top['risk_reg'], top['risk_fpca'],
        param_label='k_0',
        param_format=lambda v: f'$k_0={int(v)}$',
    )

    # Bottom row: Haar basis
    _plot_row(
        axes[1],
        bot['n_values'], bot['r2_values'],
        bot['risk_reg'], bot['risk_fpca'],
        param_label='r_2',
        param_format=lambda v: f'$r_2={v:.1f}$',
    )

    # Row labels
    axes[0][0].annotate(
        'Shifted location', xy=(-0.35, 0.5),
        xycoords='axes fraction', fontsize=12, fontweight='bold',
        rotation=90, va='center',
    )
    axes[1][0].annotate(
        'Haar basis', xy=(-0.35, 0.5),
        xycoords='axes fraction', fontsize=12, fontweight='bold',
        rotation=90, va='center',
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Figure 3 plot to {save_path}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_figure3(
        os.path.join(results_dir, 'figure3_top.npz'),
        os.path.join(results_dir, 'figure3_bottom.npz'),
        os.path.join(results_dir, 'figure3.pdf'),
    )
