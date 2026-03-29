"""Master script: run all simulations and generate figures."""

import os
import sys
import time

# Ensure we can import from the package
sys.path.insert(0, os.path.dirname(__file__))

from simulations.run_figure2 import run_figure2
from simulations.run_figure3_top import run_figure3_top
from simulations.run_figure3_bottom import run_figure3_bottom
from plots.plot_figure2 import plot_figure2
from plots.plot_figure3 import plot_figure3


def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig2_path = os.path.join(results_dir, 'figure2.npz')
    fig3t_path = os.path.join(results_dir, 'figure3_top.npz')
    fig3b_path = os.path.join(results_dir, 'figure3_bottom.npz')

    # --- Simulations ---
    print("=" * 60)
    print("Running Figure 2 simulations (aligned cosine basis)...")
    t0 = time.time()
    run_figure2(save_path=fig2_path)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Running Figure 3 top simulations (shifted k0)...")
    t0 = time.time()
    run_figure3_top(save_path=fig3t_path)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Running Figure 3 bottom simulations (Haar basis)...")
    t0 = time.time()
    run_figure3_bottom(save_path=fig3b_path)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # --- Plots ---
    print("Generating Figure 2 plot...")
    plot_figure2(fig2_path, os.path.join(results_dir, 'figure2.pdf'))

    print("Generating Figure 3 plot...")
    plot_figure3(fig3t_path, fig3b_path, os.path.join(results_dir, 'figure3.pdf'))

    print("=" * 60)
    print("All done! Check results/ for output files:")
    print(f"  {os.path.join(results_dir, 'figure2.pdf')}")
    print(f"  {os.path.join(results_dir, 'figure3.pdf')}")


if __name__ == '__main__':
    main()
