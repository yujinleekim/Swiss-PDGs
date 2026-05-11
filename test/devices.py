"""
devices.py
──────────
Device-level FOR models for the FFOR computation.

Three device types (paper Section II-A):
  1. BESSModel           – apparent power circle + SOC energy constraint
  2. PVModel             – active power ∈ [0, P_max], reactive power via PF range
  3. ControllableLoadModel – fixed power factor line + energy constraint (heat pump)

Each model exposes:
  .add_constraints(m, t, baseline)   → adds Gurobi constraints to model m
  .for_polygon(n_sides)              → returns (P_vertices, Q_vertices) of FOR polygon
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_vertices(S_max: float, n_sides: int = 8):
    """
    Regular n-gon inscribed in circle of radius S_max.
    Returns arrays of (P, Q) vertices.
    Used to linearise the circular apparent power constraint.
    """
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    P = S_max * np.cos(angles)
    Q = S_max * np.sin(angles)
    return P, Q


def _add_polygon_constraint(m: gp.Model, P_var, Q_var,
                             S_max: float, n_sides: int = 8,
                             name_prefix: str = ""):
    """
    Linearise  P² + Q² ≤ S_max²  using a regular n-gon.
    Each edge of the polygon becomes one linear inequality.
    """
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    for i in range(n_sides):
        a1 = angles[i]
        a2 = angles[(i + 1) % n_sides]
        # normal vector to edge
        nx = np.cos((a1 + a2) / 2)
        ny = np.sin((a1 + a2) / 2)
        rhs = S_max * np.cos((a2 - a1) / 2)
        m.addConstr(nx * P_var + ny * Q_var <= rhs,
                    name=f"{name_prefix}_edge_{i}")


# ─────────────────────────────────────────────────────────────────────────────
# BESS
# ─────────────────────────────────────────────────────────────────────────────

class BESSModel:
    """
    Battery Energy Storage System.

    Constraints (paper eq. 1):
        √(P_s² + Q_s²) ≤ S_max      (linearised as polygon)
        SoC_{t+1} = SoC_t − Δt·P_s / C_s
        SoC_min ≤ SoC_t ≤ SoC_max
    """

    def __init__(self, bus: int, S_max_MW: float, E_cap_MWh: float,
                 soc_min: float = 0.2, soc_max: float = 0.9,
                 soc_init: float = 0.5):
        self.bus = bus
        self.S_max = S_max_MW
        self.E_cap = E_cap_MWh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_init = soc_init

    def add_single_timestep_constraints(self, m: gp.Model,
                                        P_var, Q_var,
                                        dt_h: float = 0.25,
                                        name: str = "bess"):
        """Add BESS constraints for a single timestep (no SOC propagation)."""
        _add_polygon_constraint(m, P_var, Q_var, self.S_max,
                                n_sides=8, name_prefix=name)

        # Ensure SOC stays feasible for at least one timestep
        delta_soc_max = self.S_max * dt_h / self.E_cap
        m.addConstr(P_var <=  self.S_max, name=f"{name}_P_upper")
        m.addConstr(P_var >= -self.S_max, name=f"{name}_P_lower")
        # SOC: if P > 0 (discharging), need enough energy
        soc_avail = (self.soc_init - self.soc_min) * self.E_cap  # MWh
        m.addConstr(P_var <= soc_avail / dt_h, name=f"{name}_soc_discharge")
        soc_room  = (self.soc_max - self.soc_init) * self.E_cap
        m.addConstr(P_var >= -soc_room / dt_h, name=f"{name}_soc_charge")

    def add_multi_timestep_constraints(self, m: gp.Model,
                                       P_vars: list, Q_vars: list,
                                       soc_vars: list,
                                       dt_h: float = 0.25,
                                       name: str = "bess"):
        """Add BESS constraints over multiple timesteps with SOC coupling."""
        T = len(P_vars)
        # initial SOC
        m.addConstr(soc_vars[0] == self.soc_init, name=f"{name}_soc0")
        for t in range(T):
            _add_polygon_constraint(m, P_vars[t], Q_vars[t], self.S_max,
                                    n_sides=8, name_prefix=f"{name}_t{t}")
            m.addConstr(soc_vars[t] >= self.soc_min, name=f"{name}_socmin_t{t}")
            m.addConstr(soc_vars[t] <= self.soc_max, name=f"{name}_socmax_t{t}")
            if t < T - 1:
                m.addConstr(
                    soc_vars[t + 1] == soc_vars[t] - dt_h * P_vars[t] / self.E_cap,
                    name=f"{name}_soc_dyn_t{t}",
                )

    def for_polygon(self, n_sides: int = 8):
        return _polygon_vertices(self.S_max, n_sides)


# ─────────────────────────────────────────────────────────────────────────────
# PV
# ─────────────────────────────────────────────────────────────────────────────

class PVModel:
    """
    PV system.

    Constraints (paper Section II-A-2):
        0 ≤ P_pv ≤ P_max(t)          (curtailable active power)
        Q_pv ∈ [P_pv·tan(arccos(pf_min)), −P_pv·tan(arccos(pf_min))]
    """

    def __init__(self, bus: int, S_max_MW: float, pf_min: float = 0.9):
        self.bus = bus
        self.S_max = S_max_MW
        self.pf_min = pf_min
        self.tan_phi = np.tan(np.arccos(pf_min))

    def p_max(self, irradiance: float) -> float:
        """Maximum active power [MW] given irradiance [W/m²]."""
        return self.S_max * irradiance / 1000.0

    def add_constraints(self, m: gp.Model, P_var, Q_var,
                        irradiance: float, name: str = "pv"):
        pmax = self.p_max(irradiance)
        m.addConstr(P_var >= 0,    name=f"{name}_P_lower")
        m.addConstr(P_var <= pmax, name=f"{name}_P_upper")
        # reactive power bounded by power factor range
        m.addConstr(Q_var <=  self.tan_phi * P_var, name=f"{name}_Q_upper")
        m.addConstr(Q_var >= -self.tan_phi * P_var, name=f"{name}_Q_lower")

    def for_polygon(self, irradiance: float = 1000.0):
        pmax = self.p_max(irradiance)
        P = [0, pmax, pmax, 0]
        Q = [0,  self.tan_phi * pmax, -self.tan_phi * pmax, 0]
        return np.array(P), np.array(Q)


# ─────────────────────────────────────────────────────────────────────────────
# Controllable Load (Heat Pump / Electric Boiler)
# ─────────────────────────────────────────────────────────────────────────────

class ControllableLoadModel:
    """
    Flexible load with fixed power factor (heat pump or boiler).

    Constraints (paper Section II-A-3):
        0 ≤ P_c ≤ P_max
        Q_c = P_c · tan(arccos(pf))      (fixed PF → line in PQ plane)

    Optional thermal energy coupling (heat pump only, paper eq. 2):
        T_{t+1} = T_t + (P_flex / P_base) · q_heat · Δt
        T_min ≤ T_t ≤ T_max
    """

    def __init__(self, bus: int, P_max_MW: float, pf: float = 0.95,
                 device_type: str = "hp",
                 T_min: float = 19.0, T_max: float = 23.0,
                 T_init: float = 21.0):
        self.bus = bus
        self.P_max = P_max_MW
        self.pf = pf
        self.tan_phi = np.tan(np.arccos(pf))
        self.type = device_type
        self.T_min = T_min
        self.T_max = T_max
        self.T_init = T_init

    def add_single_timestep_constraints(self, m: gp.Model,
                                        P_var, Q_var,
                                        P_base: float = 0.0,
                                        name: str = "load"):
        m.addConstr(P_var >= 0,        name=f"{name}_P_lower")
        m.addConstr(P_var <= self.P_max, name=f"{name}_P_upper")
        # fixed power factor: Q = P · tan(φ)
        m.addConstr(Q_var == self.tan_phi * P_var, name=f"{name}_pf")

    def add_multi_timestep_constraints(self, m: gp.Model,
                                       P_vars: list, Q_vars: list,
                                       T_vars: list,
                                       P_base_series: np.ndarray,
                                       q_heat_series: np.ndarray,
                                       dt_h: float = 0.25,
                                       name: str = "load"):
        """
        Multi-timestep with optional temperature coupling (heat pumps).
        """
        T = len(P_vars)
        if self.type == "hp":
            m.addConstr(T_vars[0] == self.T_init, name=f"{name}_T0")
        for t in range(T):
            m.addConstr(P_vars[t] >= 0,           name=f"{name}_P_lower_t{t}")
            m.addConstr(P_vars[t] <= self.P_max,  name=f"{name}_P_upper_t{t}")
            m.addConstr(Q_vars[t] == self.tan_phi * P_vars[t],
                        name=f"{name}_pf_t{t}")
            if self.type == "hp" and t < T - 1:
                P_base = max(P_base_series[t], 1e-6)
                P_flex = P_vars[t] - P_base
                m.addConstr(
                    T_vars[t + 1] == T_vars[t]
                    + (P_flex / P_base) * q_heat_series[t] * dt_h,
                    name=f"{name}_T_dyn_t{t}",
                )
                m.addConstr(T_vars[t] >= self.T_min, name=f"{name}_Tmin_t{t}")
                m.addConstr(T_vars[t] <= self.T_max, name=f"{name}_Tmax_t{t}")

    def for_polygon(self, P_base: float = 0.0):
        P = np.array([0.0, self.P_max])
        Q = self.tan_phi * P
        return P, Q