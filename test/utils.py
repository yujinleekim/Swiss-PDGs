"""
utils.py
────────
Plotting helpers for FFOR results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.spatial import ConvexHull


def _close_polygon(P, Q):
    """Close a polygon by appending the first point."""
    return np.append(P, P[0]), np.append(Q, Q[0])


def _convex_hull_order(P, Q):
    """Return P, Q reordered along the convex hull boundary."""
    pts = np.column_stack([P, Q])
    if len(pts) < 3:
        return P, Q
    try:
        hull = ConvexHull(pts)
        idx = hull.vertices
        return pts[idx, 0], pts[idx, 1]
    except Exception:
        return P, Q


def plot_ffor(result, title: str = "FFOR",
              save_path: Path = None):
    """Plot a single-timestep FFOR."""
    P, Q = _convex_hull_order(result.P_flex, result.Q_flex)
    P, Q = _close_polygon(P, Q)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill(P, Q, alpha=0.25, color="steelblue", label="FFOR")
    ax.plot(P, Q, "b-", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--")
    ax.plot(0, 0, "ko", markersize=6, label="Baseline (origin)")
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ffor_multi_duration(ffor_multi: dict,
                             title: str = "Multi-Timestep FFOR",
                             save_path: Path = None):
    """
    Plot FFOR polygons for multiple sustained durations on one figure.
    Colour-coded from short (warm) to long (cool).
    """
    durations = sorted(ffor_multi.keys())
    cmap = cm.get_cmap("plasma_r", len(durations))

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, d in enumerate(durations):
        res = ffor_multi[d]
        if len(res.P_flex) < 2:
            continue
        P, Q = _convex_hull_order(res.P_flex, res.Q_flex)
        P, Q = _close_polygon(P, Q)
        label = f"{d} h" if d >= 1 else f"{int(d*60)} min"
        ax.fill(P, Q, alpha=0.15, color=cmap(i))
        ax.plot(P, Q, color=cmap(i), linewidth=1.5, label=label)

    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--")
    ax.plot(0, 0, "ko", markersize=6)
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_active_flex_vs_duration(ffor_multi: dict,
                                 save_path: Path = None):
    """
    Right panel of paper Fig. 6:
    Max positive/negative active power flexibility vs sustained duration.
    """
    durations = sorted(ffor_multi.keys())
    p_pos = [max(ffor_multi[d].P_flex, default=0) for d in durations]
    p_neg = [min(ffor_multi[d].P_flex, default=0) for d in durations]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(durations, p_pos, "b-o", label="Positive flexibility")
    ax.plot(durations, p_neg, "r-o", label="Negative flexibility")
    ax.set_xlabel("Sustained Duration [h]")
    ax.set_ylabel("Active Power Flexibility [MW]")
    ax.set_title("Flexibility vs Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)