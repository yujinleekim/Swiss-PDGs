"""
grid_model.py
─────────────
Wasterkingen LV distribution grid (Swiss-PDGs dataset, grid ID: 70-1_0_4).
  - Loads real MATPOWER CSV data; falls back to synthetic 9-bus MV network
  - Builds the linearised power-flow Jacobian  J = [JPθ JPU; JQθ JQU]
  - Auto-derives BESS / PV / controllable-load placement from the network
  - Generates synthetic baseline time series scaled to actual load level
"""

import numpy as np
import pandas as pd
import pandapower as pp
from pathlib import Path
from devices import BESSModel, PVModel, ControllableLoadModel


# ── paths to real Wasterkingen data ───────────────────────────────────────────

_HERE          = Path(__file__).parent
_REAL_MPC_DIR  = _HERE.parent / "grids/matpower_data/LV/Midlands-Periurban/70-1_0_4"
_REAL_PP_FILE  = _HERE.parent / "grids/pandapower_data/LV/Midlands-Periurban/70-1_0_4_grid.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_jacobian(net):
    """
    Linearised power-flow Jacobian around flat start (|V|=1 p.u., θ=0).

        J = | JPθ  JPU |
            | JQθ  JQU |

    Row/column order follows net.bus.index (positional order).
    """
    n = len(net.bus)
    g    = np.zeros((n, n))
    b    = np.zeros((n, n))
    b_sh = np.zeros((n, n))

    for _, row in net.line.iterrows():
        i = net.bus.index.get_loc(int(row.from_bus))
        k = net.bus.index.get_loc(int(row.to_bus))
        z = complex(row.r_ohm_per_km, row.x_ohm_per_km) * row.length_km
        if abs(z) < 1e-12:
            continue
        y   = 1.0 / z
        gik = y.real
        bik = y.imag
        bsh = (row.c_nf_per_km * row.length_km * 1e-9 * 2 * np.pi * 50) / 2

        g[i, k] -= gik;  g[k, i] -= gik
        g[i, i] += gik;  g[k, k] += gik
        b[i, k] -= bik;  b[k, i] -= bik
        b[i, i] += bik;  b[k, k] += bik
        b_sh[i, k] += bsh; b_sh[k, i] += bsh

    JPt = np.zeros((n, n))
    JPU = np.zeros((n, n))
    JQt = np.zeros((n, n))
    JQU = np.zeros((n, n))

    for i in range(n):
        for k in range(n):
            if i == k:
                JPt[i, i] = -sum(b[i, j]                     for j in range(n) if j != i)
                JQU[i, i] = -sum(2 * b_sh[i, j] + b[i, j]   for j in range(n) if j != i)
                JPU[i, i] =  sum(g[i, j]                     for j in range(n) if j != i)
                JQt[i, i] = -JPU[i, i]
            else:
                JPt[i, k] =  b[i, k]
                JQU[i, k] =  b[i, k]
                JPU[i, k] = -g[i, k]
                JQt[i, k] =  g[i, k]

    J = np.block([[JPt, JPU], [JQt, JQU]])
    return J, {"JPt": JPt, "JPU": JPU, "JQt": JQt, "JQU": JQU}


# ─────────────────────────────────────────────────────────────────────────────
# Grid class
# ─────────────────────────────────────────────────────────────────────────────

class WasterkingenGrid:
    """
    Wasterkingen distribution grid.
    Loads real LV data (70-1_0_4) when available; falls back to a synthetic
    9-bus MV network for offline/demo use.
    """

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.net      = self._load_or_build_network()
        self.n_buses  = len(self.net.bus)
        self.n_lines  = len(self.net.line)
        self.pcc_bus  = int(self.net.ext_grid.bus.iloc[0])
        self.J        = None
        self.J_blocks = None

        # Device placement derived from network topology
        self._BESS_DATA, self._PV_DATA, self._LOAD_DATA = self._make_device_data()

    # ── bus index helpers ─────────────────────────────────────────────────────

    def bus_pos(self, bus_idx: int) -> int:
        """Pandapower bus index → row/column position in the Jacobian matrix."""
        return self.net.bus.index.get_loc(bus_idx)

    @property
    def pcc_pos(self) -> int:
        """Positional index of the PCC (slack) bus in the Jacobian."""
        return self.bus_pos(self.pcc_bus)

    # ── network loading ───────────────────────────────────────────────────────

    def _load_or_build_network(self) -> pp.pandapowerNet:
        # 1. Try pandapower Excel
        if _REAL_PP_FILE.exists():
            try:
                net = pp.from_excel(str(_REAL_PP_FILE))
                print(f"  Loaded real grid: {_REAL_PP_FILE.name} "
                      f"({len(net.bus)} buses, {len(net.line)} lines)")
                return net
            except Exception as e:
                print(f"  [warn] pp.from_excel failed ({e}), trying CSV …")

        # 2. Try MATPOWER CSV
        if _REAL_MPC_DIR.exists():
            try:
                net = self._load_from_matpower_csv(_REAL_MPC_DIR)
                print(f"  Loaded real grid from MATPOWER CSV "
                      f"({len(net.bus)} buses, {len(net.line)} lines)")
                return net
            except Exception as e:
                print(f"  [warn] MATPOWER CSV load failed ({e}), using synthetic …")

        # 3. Synthetic fallback
        print("  [warn] Real data not found — using synthetic 9-bus MV network")
        return self._build_synthetic_network()

    def _load_from_matpower_csv(self, csv_dir: Path) -> pp.pandapowerNet:
        name      = csv_dir.name
        bus_df    = pd.read_csv(csv_dir / f"{name}_bus_data.csv")
        branch_df = pd.read_csv(csv_dir / f"{name}_branch_data.csv")

        vn_kv  = float(bus_df["baseKV"].iloc[0])   # 0.4 kV for LV
        sn_mva = 100.0                               # MATPOWER default base
        z_base = vn_kv ** 2 / sn_mva               # Ohm

        net = pp.create_empty_network(f_hz=50, sn_mva=sn_mva)

        # Buses
        bus_map = {}
        for _, row in bus_df.iterrows():
            idx = pp.create_bus(net, vn_kv=vn_kv,
                                name=f"Bus_{int(row.BUS_I)}",
                                max_vm_pu=float(row.Vmax),
                                min_vm_pu=float(row.Vmin))
            bus_map[int(row.BUS_I)] = idx

        # External grid (slack bus)
        for _, row in bus_df[bus_df["BUS_TYPE"] == 3].iterrows():
            pp.create_ext_grid(net, bus=bus_map[int(row.BUS_I)],
                               vm_pu=1.0, va_degree=0.0,
                               name="HV/LV Transformer")

        # Loads
        for _, row in bus_df.iterrows():
            if float(row.Pd) > 0 or float(row.Qd) > 0:
                pp.create_load(net, bus=bus_map[int(row.BUS_I)],
                               p_mw=float(row.Pd), q_mvar=float(row.Qd),
                               name=f"Load_{int(row.BUS_I)}")

        # Lines (impedance in p.u. on 100-MVA base → convert to Ohm)
        for _, row in branch_df.iterrows():
            if int(row.BR_STATUS) == 0:
                continue
            r    = float(row.BR_R) * z_base
            x    = float(row.BR_X) * z_base
            b_pu = float(row.BR_B)
            c_nf = b_pu / (2 * np.pi * 50 * z_base) * 1e9 if b_pu > 0 else 0.0
            pp.create_line_from_parameters(
                net,
                from_bus=bus_map[int(row.F_BUS)],
                to_bus=bus_map[int(row.T_BUS)],
                length_km=1.0,          # length absorbed into r/x totals
                r_ohm_per_km=r,
                x_ohm_per_km=x,
                c_nf_per_km=c_nf,
                max_i_ka=0.2,
                name=f"Line_{int(row.F_BUS)}-{int(row.T_BUS)}",
            )

        return net

    def _build_synthetic_network(self) -> pp.pandapowerNet:
        """9-bus MV synthetic fallback (original sketch)."""
        net = pp.create_empty_network(f_hz=50, sn_mva=10)
        vn  = 16.0
        for i in range(9):
            pp.create_bus(net, vn_kv=vn, name=f"Bus {i}")
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0,
                           name="HV/MV Transformer")
        pp.create_sgen(net, bus=1, p_mw=0.5, q_mvar=0.0, name="Gen1_RoR")
        pp.create_sgen(net, bus=2, p_mw=0.3, q_mvar=0.0, name="Gen2_RoR")
        pp.create_sgen(net, bus=3, p_mw=0.0, q_mvar=0.0, name="BESS")
        pp.create_load(net, bus=4, p_mw=1.0, q_mvar=0.10, name="Load_A")
        pp.create_load(net, bus=5, p_mw=0.7, q_mvar=0.07, name="Load_B")
        pp.create_load(net, bus=6, p_mw=0.5, q_mvar=0.05, name="Load_C")
        pp.create_load(net, bus=7, p_mw=0.8, q_mvar=0.08, name="Load_D")
        pp.create_load(net, bus=8, p_mw=0.4, q_mvar=0.04, name="Load_E")
        for fb, tb, length in [(0,1,2.0),(0,4,3.0),(1,2,1.5),(1,5,2.5),
                                (2,3,1.0),(3,6,2.0),(4,7,1.8),(5,8,2.2)]:
            pp.create_line_from_parameters(
                net, from_bus=fb, to_bus=tb, length_km=length,
                r_ohm_per_km=0.5, x_ohm_per_km=0.4,
                c_nf_per_km=10.0, max_i_ka=0.2,
                name=f"Line_{fb}-{tb}")
        return net

    # ── device placement ──────────────────────────────────────────────────────

    def _make_device_data(self):
        """
        Derive device placement from the loaded network.

        LV real grid: selects top-load buses; scales device ratings to LV level.
        MV synthetic:  returns the original hardcoded values.
        """
        is_lv = self.net.bus.vn_kv.max() < 1.0

        if not is_lv:
            bess = [{"bus": 3, "S_max_MW": 4.0, "E_cap_MWh": 5.0,
                     "soc_min": 0.4, "soc_max": 0.8, "soc_init": 0.6}]
            pv   = [{"bus": 4, "S_max_MW": 0.5, "pf_min": 0.9},
                    {"bus": 5, "S_max_MW": 0.3, "pf_min": 0.9},
                    {"bus": 6, "S_max_MW": 0.8, "pf_min": 0.9},
                    {"bus": 7, "S_max_MW": 0.2, "pf_min": 0.9}]
            load = [{"bus": 4, "P_max_MW": 1.2, "pf": 0.95, "device_type": "hp"},
                    {"bus": 5, "P_max_MW": 0.8, "pf": 0.95, "device_type": "hp"},
                    {"bus": 6, "P_max_MW": 0.6, "pf": 0.95, "device_type": "boiler"},
                    {"bus": 7, "P_max_MW": 0.9, "pf": 0.95, "device_type": "hp"},
                    {"bus": 8, "P_max_MW": 0.5, "pf": 0.95, "device_type": "hp"}]
            return bess, pv, load

        # Real LV: sort load buses by peak demand
        load_df = (self.net.load[self.net.load.bus != self.pcc_bus]
                   .copy()
                   .sort_values("p_mw", ascending=False)
                   .reset_index(drop=True))

        # BESS at the bus with the largest single-customer load
        top_bus   = int(load_df.iloc[0].bus)
        p_top     = float(load_df.iloc[0].p_mw)
        bess = [{"bus": top_bus,
                 "S_max_MW":  round(p_top * 5,  5),
                 "E_cap_MWh": round(p_top * 10, 5),
                 "soc_min": 0.2, "soc_max": 0.9, "soc_init": 0.5}]

        # PV at top-4 load buses (rooftop PV oversized for flexibility)
        pv = [{"bus": int(load_df.iloc[i].bus),
               "S_max_MW": round(float(load_df.iloc[i].p_mw) * 3, 5),
               "pf_min": 0.9}
              for i in range(min(4, len(load_df)))]

        # Controllable loads: next 5 buses (index 4–8)
        load = [{"bus": int(load_df.iloc[i].bus),
                 "P_max_MW": round(float(load_df.iloc[i].p_mw) * 1.5, 5),
                 "pf": 0.95, "device_type": "hp"}
                for i in range(min(4, len(load_df)), min(9, len(load_df)))]

        return bess, pv, load

    # ── Jacobian ──────────────────────────────────────────────────────────────

    def build_jacobian(self):
        self.J, self.J_blocks = _build_jacobian(self.net)
        return self.J

    # ── device accessors ──────────────────────────────────────────────────────

    def get_bess_devices(self):
        return [BESSModel(**d) for d in self._BESS_DATA]

    def get_pv_devices(self):
        return [PVModel(**d) for d in self._PV_DATA]

    def get_controllable_loads(self):
        return [ControllableLoadModel(**d) for d in self._LOAD_DATA]

    # ── baseline time series ──────────────────────────────────────────────────

    def get_baseline(self, date: str = "2021-09-01",
                     resolution_min: int = 15) -> pd.DataFrame:
        csv_file = self.data_dir / f"baseline_{date}.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file, parse_dates=["time"])
        return self._synthetic_baseline(date, resolution_min)

    def _synthetic_baseline(self, date: str, resolution_min: int) -> pd.DataFrame:
        n_steps     = 24 * 60 // resolution_min
        t           = np.linspace(0, 24, n_steps, endpoint=False)
        irradiance  = np.clip(800 * np.exp(-0.5 * ((t - 12) / 3) ** 2), 0, None)
        temperature = 15 + 8 * np.sin(2 * np.pi * (t - 6) / 24)

        rows = []
        for ti, irr, temp in zip(t, irradiance, temperature):
            row = {"time": ti, "irradiance": irr, "temperature": temp}

            # PV baselines
            for d in self._PV_DATA:
                b = d["bus"]
                row[f"bus_{b}_P_pv_base"] = d["S_max_MW"] * irr / 1000.0
                row[f"bus_{b}_Q_pv_base"] = 0.0

            # Load (HP) baselines
            for d in self._LOAD_DATA:
                b       = d["bus"]
                tan_phi = np.tan(np.arccos(d["pf"]))
                if d["device_type"] == "hp":
                    p_hp = d["P_max_MW"] * max(0, (15 - temp) / 23)
                else:
                    p_hp = d["P_max_MW"] * 0.3
                row[f"bus_{b}_P_hp_base"] = p_hp
                row[f"bus_{b}_Q_hp_base"] = p_hp * tan_phi

            rows.append(row)

        return pd.DataFrame(rows)

    # ── line / voltage limits ─────────────────────────────────────────────────

    @property
    def line_ratings_mva(self) -> np.ndarray:
        vn = self.net.bus.vn_kv.iloc[0]
        return self.net.line.max_i_ka.values * vn * np.sqrt(3)

    @property
    def v_limits(self) -> tuple:
        """Voltage band: ±10 % for LV, ±5 % for MV."""
        is_lv = self.net.bus.vn_kv.max() < 1.0
        return (0.90, 1.10) if is_lv else (0.95, 1.05)
