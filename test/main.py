"""
FFOR (Feasible Flexibility Operating Region) Computation
for the Wasterkingen Distribution Grid

Based on:
  "On the Flexibility Potential of a Swiss Distribution Grid:
   Opportunities and Limitations" (Brändle et al., 2025)

Group: Kim, Ng, Laser
Course: Optimization in Energy Systems 2026, ETH Zürich
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from grid_model import WasterkingenGrid
from devices import BESSModel, PVModel, ControllableLoadModel
from ffor_optimizer import FFOROptimizer
from utils import plot_ffor, plot_ffor_multi_duration

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("FFOR Computation – Wasterkingen Distribution Grid")
    print("=" * 60)

    # ── 1. Load grid ─────────────────────────────────────────────
    print("\n[1/4] Loading grid topology ...")
    grid = WasterkingenGrid(data_dir=DATA_DIR)
    grid.build_jacobian()
    print(f"      Nodes  : {grid.n_buses}")
    print(f"      Lines  : {grid.n_lines}")
    print(f"      PCC bus: {grid.pcc_bus}")

    # ── 2. Instantiate flexible devices ──────────────────────────
    print("\n[2/4] Setting up flexible devices ...")
    bess_list = grid.get_bess_devices()
    pv_list   = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()
    print(f"      BESS  : {len(bess_list)} units")
    print(f"      PV    : {len(pv_list)} units")
    print(f"      Loads : {len(load_list)} units")

    # ── 3. Load time-series baseline ─────────────────────────────
    print("\n[3/4] Loading baseline time series ...")
    baseline = grid.get_baseline(
        date="2021-09-01",
        resolution_min=15,
    )
    print(f"      Timesteps: {len(baseline)}")

    # ── 4. Compute FFOR ──────────────────────────────────────────
    print("\n[4/4] Computing FFOR ...")
    optimizer = FFOROptimizer(grid, bess_list, pv_list, load_list)

    # --- single-timestep FFOR at 14:00 ---
    t_idx = 14 * 4          # 14:00 with 15-min resolution → index 56
    ffor_single = optimizer.compute_ffor_single_timestep(
        t_idx=t_idx,
        baseline=baseline,
        n_directions=36,    # 10° increments
    )

    # --- multi-timestep FFOR for several sustained durations ---
    durations_h = [0.25, 0.5, 1, 2, 4, 8]
    ffor_multi = {}
    for d in durations_h:
        print(f"   Duration {d} h ...")
        ffor_multi[d] = optimizer.compute_ffor_multi_timestep(
            t_start=t_idx,
            duration_h=d,
            baseline=baseline,
            n_directions=36,
        )

    # ── Plots ─────────────────────────────────────────────────────
    print("\nPlotting results ...")
    plot_ffor(
        ffor_single,
        title="Single-Timestep FFOR – Wasterkingen, 01-Sep-2021 14:00",
        save_path=OUTPUT_DIR / "ffor_single.png",
    )
    plot_ffor_multi_duration(
        ffor_multi,
        title="Multi-Timestep FFOR – Wasterkingen, 01-Sep-2021 14:00",
        save_path=OUTPUT_DIR / "ffor_multi.png",
    )

    print(f"\nDone. Figures saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()