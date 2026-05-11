"""
ffor_optimizer.py
─────────────────
FFOROptimizer: direction-scanning LP to compute the FFOR at the PCC.

For each direction θ ∈ [0, 2π):
    maximise  cos(θ)·ΔP_pcc + sin(θ)·ΔQ_pcc
    s.t.      device operating constraints
              linearised power-flow voltage bounds (via J_red⁻¹)

Generalised for any PCC bus position (not assumed to be bus 0).
"""

from dataclasses import dataclass
import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class FFORResult:
    P_flex: np.ndarray   # boundary active power points [MW]
    Q_flex: np.ndarray   # boundary reactive power points [MVAr]


class FFOROptimizer:

    def __init__(self, grid, bess_list, pv_list, load_list):
        self.grid      = grid
        self.bess_list = bess_list
        self.pv_list   = pv_list
        self.load_list = load_list

        # Pre-compute reduced Jacobian inverse once
        J_red        = self._reduced_jacobian()
        self._J_inv  = np.linalg.pinv(J_red)   # (2*(n-1)) × (2*(n-1))
        self._n_pq   = grid.n_buses - 1

        # Non-PCC positional indices (same ordering as J_red rows/cols)
        s = grid.pcc_pos
        self._non_pcc = [i for i in range(grid.n_buses) if i != s]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reduced_jacobian(self):
        """Remove the PCC (slack) row and column from J."""
        n = self.grid.n_buses
        J = self.grid.J
        s = self.grid.pcc_pos
        keep = [i for i in range(n) if i != s] + [n + i for i in range(n) if i != s]
        return J[np.ix_(keep, keep)]

    def _irr(self, baseline, t):
        return float(baseline.iloc[min(t, len(baseline) - 1)].get("irradiance", 0.0))

    def _p_base_load(self, ld, baseline, t):
        return float(baseline.iloc[min(t, len(baseline) - 1)].get(
            f"bus_{ld.bus}_P_hp_base", 0.0))

    def _injection_deviations(self, bess_P, bess_Q, pv_P, pv_Q,
                               load_P, load_Q, baseline, t_abs):
        """
        Build {positional_index: LinExpr} for ΔP and ΔQ deviations from baseline.
        Keys are positional indices, excluding the PCC bus.
        Sign: generation positive, consumption negative.
        """
        row = baseline.iloc[min(t_abs, len(baseline) - 1)]
        dP  = {pos: gp.LinExpr() for pos in self._non_pcc}
        dQ  = {pos: gp.LinExpr() for pos in self._non_pcc}

        for i, bess in enumerate(self.bess_list):
            pos = self.grid.bus_pos(bess.bus)
            dP[pos] += bess_P[i]
            dQ[pos] += bess_Q[i]

        for i, pv in enumerate(self.pv_list):
            pos = self.grid.bus_pos(pv.bus)
            dP[pos] += pv_P[i] - float(row.get(f"bus_{pv.bus}_P_pv_base", 0.0))
            dQ[pos] += pv_Q[i] - float(row.get(f"bus_{pv.bus}_Q_pv_base", 0.0))

        for i, ld in enumerate(self.load_list):
            pos = self.grid.bus_pos(ld.bus)
            pb  = self._p_base_load(ld, baseline, t_abs)
            dP[pos] += -(load_P[i] - pb)
            dQ[pos] += -(load_Q[i] - pb * ld.tan_phi)

        return dP, dQ

    def _add_pcc_balance(self, m, dP, dQ, dP_pcc, dQ_pcc, tag=""):
        """Power balance: ΔP_pcc = −Σ ΔP_inj at all non-PCC buses."""
        m.addConstr(
            dP_pcc == -gp.quicksum(dP[pos] for pos in self._non_pcc),
            name=f"Pbal{tag}")
        m.addConstr(
            dQ_pcc == -gp.quicksum(dQ[pos] for pos in self._non_pcc),
            name=f"Qbal{tag}")

    def _add_voltage_constraints(self, m, dP, dQ, tag=""):
        """
        Voltage bound via pre-inverted Jacobian (no extra state variables):
            ΔU_b = J_inv[n_pq + b, :] · [ΔP_inj; ΔQ_inj]  ∈ [v_min−1, v_max−1]
        Adds 2 * n_pq inequality constraints per call.
        """
        n_pq = self._n_pq
        v_min, v_max = self.grid.v_limits
        lo = v_min - 1.0   # e.g. −0.10 for LV, −0.05 for MV
        hi = v_max - 1.0   # e.g. +0.10 for LV, +0.05 for MV

        inj  = ([dP[pos] for pos in self._non_pcc] +
                [dQ[pos] for pos in self._non_pcc])
        J_U  = self._J_inv[n_pq:]   # voltage sensitivity rows

        for b in range(n_pq):
            expr = gp.quicksum(float(J_U[b, j]) * inj[j]
                               for j in range(2 * n_pq))
            m.addConstr(expr >= lo, name=f"Vlo{b}{tag}")
            m.addConstr(expr <= hi, name=f"Vhi{b}{tag}")

    # ── single-timestep LP ────────────────────────────────────────────────────

    def _build_single_lp(self, t_idx, baseline, cos_th, sin_th,
                          enforce_voltage=True):
        m   = gp.Model()
        m.setParam("OutputFlag", 0)
        dt_h = 0.25
        irr  = self._irr(baseline, t_idx)

        bess_P, bess_Q = [], []
        for i, bess in enumerate(self.bess_list):
            p = m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}P")
            q = m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}Q")
            bess.add_single_timestep_constraints(m, p, q, dt_h=dt_h, name=f"b{i}")
            bess_P.append(p); bess_Q.append(q)

        pv_P, pv_Q = [], []
        for i, pv in enumerate(self.pv_list):
            p = m.addVar(lb=0, ub=pv.p_max(irr), name=f"pv{i}P")
            q = m.addVar(lb=-pv.S_max * pv.tan_phi,
                         ub=pv.S_max * pv.tan_phi, name=f"pv{i}Q")
            pv.add_constraints(m, p, q, irradiance=irr, name=f"pv{i}")
            pv_P.append(p); pv_Q.append(q)

        load_P, load_Q = [], []
        for i, ld in enumerate(self.load_list):
            p = m.addVar(lb=0, ub=ld.P_max, name=f"l{i}P")
            q = m.addVar(lb=0, ub=ld.P_max * ld.tan_phi, name=f"l{i}Q")
            ld.add_single_timestep_constraints(
                m, p, q, P_base=self._p_base_load(ld, baseline, t_idx),
                name=f"l{i}")
            load_P.append(p); load_Q.append(q)

        m.update()

        dP, dQ = self._injection_deviations(
            bess_P, bess_Q, pv_P, pv_Q, load_P, load_Q, baseline, t_idx)

        dP_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dP_pcc")
        dQ_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dQ_pcc")
        self._add_pcc_balance(m, dP, dQ, dP_pcc, dQ_pcc)
        if enforce_voltage:
            self._add_voltage_constraints(m, dP, dQ)

        m.setObjective(cos_th * dP_pcc + sin_th * dQ_pcc, GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            return dP_pcc.X, dQ_pcc.X
        return np.nan, np.nan

    def compute_ffor_single_timestep(self, t_idx, baseline,
                                     n_directions: int = 36) -> FFORResult:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        pts    = [self._build_single_lp(t_idx, baseline,
                                        np.cos(a), np.sin(a),
                                        enforce_voltage=True)
                  for a in angles]

        if all(np.isnan(p) for p, _ in pts):
            print("  [warn] voltage constraints infeasible — re-solving without")
            pts = [self._build_single_lp(t_idx, baseline,
                                          np.cos(a), np.sin(a),
                                          enforce_voltage=False)
                   for a in angles]

        pts = [(p, q) for p, q in pts if not np.isnan(p)]
        if not pts:
            return FFORResult(np.array([]), np.array([]))
        P, Q = zip(*pts)
        return FFORResult(np.array(P), np.array(Q))

    # ── multi-timestep LP ─────────────────────────────────────────────────────

    def _build_multi_lp(self, t_start, T, baseline, dt_h,
                         cos_th, sin_th, enforce_voltage=True):
        """
        Sustained-flexibility LP over T timesteps.
        BESS: full SOC dynamics.
        PV:   per-step irradiance bounds.
        Loads: simple per-step bounds (no thermal state; keeps model compact).
        Voltage: J_inv inequalities — no extra state variables.
        """
        m = gp.Model()
        m.setParam("OutputFlag", 0)

        # BESS — multi-step with SOC dynamics
        bess_Pall, bess_Qall = [], []
        for i, bess in enumerate(self.bess_list):
            Pv = [m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}P{t}") for t in range(T)]
            Qv = [m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}Q{t}") for t in range(T)]
            sv = [m.addVar(lb=bess.soc_min, ub=bess.soc_max, name=f"b{i}s{t}") for t in range(T)]
            bess.add_multi_timestep_constraints(m, Pv, Qv, sv, dt_h=dt_h, name=f"b{i}")
            bess_Pall.append(Pv); bess_Qall.append(Qv)

        # PV — per-step irradiance
        pv_Pall, pv_Qall = [], []
        for i, pv in enumerate(self.pv_list):
            Pv, Qv = [], []
            for t in range(T):
                irr = self._irr(baseline, t_start + t)
                p = m.addVar(lb=0, ub=pv.p_max(irr), name=f"pv{i}P{t}")
                q = m.addVar(lb=-pv.S_max * pv.tan_phi,
                             ub=pv.S_max * pv.tan_phi, name=f"pv{i}Q{t}")
                pv.add_constraints(m, p, q, irradiance=irr, name=f"pv{i}t{t}")
                Pv.append(p); Qv.append(q)
            pv_Pall.append(Pv); pv_Qall.append(Qv)

        # Loads — simple per-step bounds
        load_Pall, load_Qall = [], []
        for i, ld in enumerate(self.load_list):
            Pv, Qv = [], []
            for t in range(T):
                p = m.addVar(lb=0, ub=ld.P_max, name=f"l{i}P{t}")
                q = m.addVar(lb=0, ub=ld.P_max * ld.tan_phi, name=f"l{i}Q{t}")
                ld.add_single_timestep_constraints(
                    m, p, q,
                    P_base=self._p_base_load(ld, baseline, t_start + t),
                    name=f"l{i}t{t}")
                Pv.append(p); Qv.append(q)
            load_Pall.append(Pv); load_Qall.append(Qv)

        m.update()

        # Single (dP_pcc, dQ_pcc) held constant across all T timesteps
        dP_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dP_pcc")
        dQ_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dQ_pcc")

        for t in range(T):
            t_abs = t_start + t
            dP, dQ = self._injection_deviations(
                [bess_Pall[i][t] for i in range(len(self.bess_list))],
                [bess_Qall[i][t] for i in range(len(self.bess_list))],
                [pv_Pall[i][t]   for i in range(len(self.pv_list))],
                [pv_Qall[i][t]   for i in range(len(self.pv_list))],
                [load_Pall[i][t] for i in range(len(self.load_list))],
                [load_Qall[i][t] for i in range(len(self.load_list))],
                baseline, t_abs,
            )
            self._add_pcc_balance(m, dP, dQ, dP_pcc, dQ_pcc, tag=str(t))
            if enforce_voltage:
                self._add_voltage_constraints(m, dP, dQ, tag=f"t{t}")

        m.setObjective(cos_th * dP_pcc + sin_th * dQ_pcc, GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            return dP_pcc.X, dQ_pcc.X
        return np.nan, np.nan

    def compute_ffor_multi_timestep(self, t_start, duration_h, baseline,
                                    n_directions: int = 36) -> FFORResult:
        dt_h   = 0.25
        T      = max(1, round(duration_h / dt_h))
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

        pts = [self._build_multi_lp(t_start, T, baseline, dt_h,
                                    np.cos(a), np.sin(a),
                                    enforce_voltage=True)
               for a in angles]

        if all(np.isnan(p) for p, _ in pts):
            print(f"  [warn] voltage constraints infeasible for T={T} — re-solving without")
            pts = [self._build_multi_lp(t_start, T, baseline, dt_h,
                                        np.cos(a), np.sin(a),
                                        enforce_voltage=False)
                   for a in angles]

        pts = [(p, q) for p, q in pts if not np.isnan(p)]
        if not pts:
            return FFORResult(np.array([]), np.array([]))
        P, Q = zip(*pts)
        return FFORResult(np.array(P), np.array(Q))
