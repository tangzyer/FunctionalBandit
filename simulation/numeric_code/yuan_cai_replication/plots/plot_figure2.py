"""Plot Figure 2: Roughness regularization vs FPCA (aligned cosine basis)."""

import os
import numpy as np
import matplotlib.pyplot as plt


MARKERS = ['o', 's', '^', 'D', 'v']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def plot_figure2(results_path, save_path=None):
    """Create 2-panel Figure 2.

    Left: Roughness Reg excess risk vs n
    Right: FPCA excess risk vs n
    Both in log-log scale with ±1 SE error bars.
    """
    data = np.load(results_path)
    n_values = data['n_values']
    r2_values = data['r2_values']
    risk_reg = data['risk_reg']    # (n_r2, n_n, n_rep)
    risk_fpca = data['risk_fpca']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for panel_idx, (risk_data, title) in enumerate([
        (risk_reg, 'Roughness Regularization'),
        (risk_fpca, 'FPCA'),
    ]):
        ax = axes[panel_idx]
        for i, r2 in enumerate(r2_values):
            means = np.mean(risk_data[i], axis=1)
            se = np.std(risk_data[i], axis=1) / np.sqrt(risk_data.shape[2])

            ax.errorbar(
                n_values, means, yerr=se,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f'$r_2={r2:.1f}$',
                capsize=3, linewidth=1.5, markersize=5,
            )

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('n (sample size)')
        ax.set_ylabel('Excess Risk')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Figure 2 plot to {save_path}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_figure2(
        os.path.join(results_dir, 'figure2.npz'),
        os.path.join(results_dir, 'figure2.pdf'),
    )
