"""
Scenario-based SMPC for Hall-thruster thrust control using CasADi.

This script follows the scenario-based SMPC structure from the uploaded paper:
- sample S uncertainty scenarios,
- optimize one shared input sequence across all scenarios,
- propagate one state trajectory per scenario,
- minimize the summed expected stage cost over the horizon,
- apply only the first control input and repeat.

Important scope note
--------------------
This is NOT a full HallThruster.jl PDE solver. It is a reduced-order,
control-oriented surrogate inspired by HallThruster.jl's quasineutral
1D fluid formulation. The surrogate keeps the user-requested dummy
state-estimate quantities:

    x = [nu_anom, I_d, mdot_a, B_peak]

and computes thrust from them using physically-inspired algebraic
relations based on HallThruster.jl concepts such as:
- total electron collision frequency: nu_e = nu_class + nu_anom
- Hall parameter: Omega_e = e B / (m_e nu_e)
- cross-field mobility: mu_perp = e / (m_e nu_e * (1 + Omega_e^2))
- discharge-current dependence on mobility and propellant flow
- thrust dependence on mass flow and an effective ion exhaust speed

This makes the script useful for controller prototyping when you only
have estimated lumped states or a digital twin, but it should not be
mistaken for a validated thruster-discharge simulation.

Requirements
------------
    pip install casadi numpy

Optional for plotting:
    pip install matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import casadi as ca


# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19      # C
M_E = 9.1093837015e-31          # kg
K_B = 1.380649e-23              # J/K
AMU = 1.66053906660e-27         # kg
M_XE = 131.293 * AMU            # kg, xenon ion mass
G0 = 9.80665                    # m/s^2


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ThrusterNominal:
    """Nominal operating point and surrogate-model constants."""

    A_ch: float = 2.5e-3                # m^2, effective channel area (lumped)
    L_ch: float = 0.025                 # m, channel length
    V_d_nom: float = 300.0              # V, nominal discharge voltage
    mdot_nom: float = 5.0e-6            # kg/s
    B_nom: float = 0.020                # T
    I_d_nom: float = 4.5                # A
    nu_class_nom: float = 5.0e7         # 1/s, nominal classical collision freq
    n_e_nom: float = 1.2e18             # 1/m^3, lumped electron density
    eta_acc_nom: float = 0.72           # acceleration/utilization surrogate factor
    beta_bohm_nom: float = 0.012        # nu_anom ~ beta * omega_ce
    thrust_nom: float = 0.085           # N, rough operating-point thrust

    # Dynamic time constants for the reduced-order state model.
    tau_nu: float = 2.5e-4              # s
    tau_Id: float = 4.0e-4              # s
    tau_mdot: float = 3.0e-4            # s
    tau_B: float = 2.0e-4               # s

    # Surrogate gains used to shape lumped dynamics.
    k_Id_mdot: float = 4.0e5            # A per (kg/s)
    k_Id_mu: float = 3.0e11             # A per (m^2/(V s))
    k_eta_mu: float = 2.5e8             # dimensionless gain on mobility
    k_eta_Id: float = -0.035            # dimensionless gain on discharge current


@dataclass
class SMPCConfig:
    """Controller configuration."""

    dt: float = 1.0e-4                  # s
    horizon: int = 15
    num_scenarios: int = 12
    seed: int = 7

    # Control variables are absolute commands, not increments.
    # u = [mdot_cmd, B_cmd, Vd_cmd]
    u_min: Tuple[float, float, float] = (2.0e-6, 0.012, 220.0)
    u_max: Tuple[float, float, float] = (9.0e-6, 0.035, 420.0)

    du_max: Tuple[float, float, float] = (4.0e-7, 1.5e-3, 12.0)

    # Stage cost weights
    w_thrust: float = 5.0e5
    w_current: float = 5.0
    w_input: float = 2.0e3
    w_rate: float = 5.0e4
    w_state: float = 2.0


@dataclass
class ScenarioParams:
    """Uncertain parameters sampled for each scenario."""

    beta_bohm: float
    nu_class: float
    n_e: float
    eta_acc_bias: float
    Id_bias: float


# -----------------------------------------------------------------------------
# HallThruster.jl-inspired algebraic helpers
# -----------------------------------------------------------------------------
def omega_ce(B: ca.MX | ca.SX | float) -> ca.MX | ca.SX | float:
    """Electron cyclotron frequency."""
    return E_CHARGE * ca.fabs(B) / M_E


def hall_parameter(B: ca.MX | ca.SX | float,
                   nu_e: ca.MX | ca.SX | float) -> ca.MX | ca.SX | float:
    """Omega_e = omega_ce / nu_e."""
    return omega_ce(B) / ca.fmax(nu_e, 1.0)


def mu_perp(B: ca.MX | ca.SX | float,
            nu_e: ca.MX | ca.SX | float) -> ca.MX | ca.SX | float:
    """
    HallThruster.jl-style cross-field mobility.

    mu_perp = e / (m_e * nu_e) * 1 / (1 + Omega_e^2)
    """
    Om = hall_parameter(B, nu_e)
    return (E_CHARGE / (M_E * ca.fmax(nu_e, 1.0))) / (1.0 + Om**2)


def smooth_clip(x: ca.MX | ca.SX | float,
                xmin: float,
                xmax: float,
                k: float = 20.0) -> ca.MX | ca.SX | float:
    """Smoothly squash x into [xmin, xmax] for symbolic expressions."""
    xc = 0.5 * (xmin + xmax)
    xr = 0.5 * (xmax - xmin)
    return xc + xr * ca.tanh(k * (x - xc) / max(xr, 1e-12))


def scenario_dynamics_symbolic(
    x: ca.MX,
    u: ca.MX,
    p: ScenarioParams,
    nominal: ThrusterNominal,
    dt: float,
) -> Tuple[ca.MX, ca.MX, ca.MX, ca.MX, ca.MX]:
    """
    Reduced-order dynamics inspired by HallThruster.jl quantities.

    States
    ------
    x[0] = nu_anom [1/s]
    x[1] = I_d     [A]
    x[2] = mdot_a  [kg/s]
    x[3] = B_peak  [T]

    Controls
    --------
    u[0] = mdot_cmd [kg/s]
    u[1] = B_cmd    [T]
    u[2] = Vd_cmd   [V]

    Outputs returned alongside x_next
    ---------------------------------
    thrust [N], nu_e [1/s], mu_perp [m^2/(V s)], eta_acc [-]
    """
    nu_anom, I_d, mdot_a, B_peak = x[0], x[1], x[2], x[3]
    mdot_cmd, B_cmd, Vd_cmd = u[0], u[1], u[2]

    # HallThruster.jl-style anomalous transport anchor:
    # nu_anom ~ beta_bohm * omega_ce(B)
    nu_anom_target = p.beta_bohm * omega_ce(B_peak)

    # First-order actuator/state dynamics.
    nu_anom_next = nu_anom + dt / nominal.tau_nu * (nu_anom_target - nu_anom)
    mdot_next = mdot_a + dt / nominal.tau_mdot * (mdot_cmd - mdot_a)
    B_next = B_peak + dt / nominal.tau_B * (B_cmd - B_peak)

    # Total collision frequency and mobility from HallThruster.jl relations.
    nu_e_total = p.nu_class + ca.fmax(nu_anom_next, 1.0)
    mu_cross = mu_perp(B_next, nu_e_total)

    # Lumped discharge current target inspired by the discharge-current / mobility
    # relation. This is not the full spatial integral used in HallThruster.jl.
    Id_target = (
        nominal.I_d_nom
        + nominal.k_Id_mdot * (mdot_next - nominal.mdot_nom)
        + nominal.k_Id_mu * (mu_cross - mu_perp(nominal.B_nom, nominal.nu_class_nom + nominal.beta_bohm_nom * (E_CHARGE * nominal.B_nom / M_E)))
        + 0.010 * (Vd_cmd - nominal.V_d_nom)
        + p.Id_bias
    )
    Id_next = I_d + dt / nominal.tau_Id * (Id_target - I_d)

    # Effective acceleration/utilization proxy.
    eta_raw = (
        nominal.eta_acc_nom
        + nominal.k_eta_mu * (mu_cross - mu_perp(nominal.B_nom, nominal.nu_class_nom + nominal.beta_bohm_nom * (E_CHARGE * nominal.B_nom / M_E)))
        + nominal.k_eta_Id * (Id_next - nominal.I_d_nom)
        + p.eta_acc_bias
    )
    eta_acc = smooth_clip(eta_raw, 0.30, 0.90)

    # Thrust surrogate using mdot * v_e with an effective ion exhaust speed.
    # v_e ~ sqrt(2 e eta_acc Vd / m_i)
    v_eff = ca.sqrt(ca.fmax(2.0 * E_CHARGE * eta_acc * Vd_cmd / M_XE, 1.0))
    thrust = mdot_next * v_eff

    x_next = ca.vertcat(nu_anom_next, Id_next, mdot_next, B_next)
    return x_next, thrust, nu_e_total, mu_cross, eta_acc


# -----------------------------------------------------------------------------
# Scenario sampling
# -----------------------------------------------------------------------------
def sample_scenarios(
    nominal: ThrusterNominal,
    cfg: SMPCConfig,
    uncertainty_frac: Dict[str, float] | None = None,
) -> List[ScenarioParams]:
    """Sample uncertain parameters for scenario-based SMPC."""
    if uncertainty_frac is None:
        uncertainty_frac = {
            "beta_bohm": 0.25,
            "nu_class": 0.20,
            "n_e": 0.10,
            "eta_acc_bias": 0.08,
            "Id_bias": 0.40,
        }

    rng = np.random.default_rng(cfg.seed)
    scenarios: List[ScenarioParams] = []

    def sample_uniform(center: float, frac: float) -> float:
        lo = center * (1.0 - frac)
        hi = center * (1.0 + frac)
        return float(rng.uniform(lo, hi))

    for _ in range(cfg.num_scenarios):
        scenarios.append(
            ScenarioParams(
                beta_bohm=sample_uniform(nominal.beta_bohm_nom, uncertainty_frac["beta_bohm"]),
                nu_class=sample_uniform(nominal.nu_class_nom, uncertainty_frac["nu_class"]),
                n_e=sample_uniform(nominal.n_e_nom, uncertainty_frac["n_e"]),
                eta_acc_bias=float(rng.uniform(-uncertainty_frac["eta_acc_bias"], uncertainty_frac["eta_acc_bias"])),
                Id_bias=float(rng.uniform(-uncertainty_frac["Id_bias"], uncertainty_frac["Id_bias"])),
            )
        )

    return scenarios


# -----------------------------------------------------------------------------
# SMPC controller
# -----------------------------------------------------------------------------
class HallThrusterSMPC:
    """Scenario-based SMPC controller using one shared input sequence."""

    def __init__(self, nominal: ThrusterNominal, cfg: SMPCConfig):
        self.nominal = nominal
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 123)

    def solve(
        self,
        x_est: np.ndarray,
        u_prev: np.ndarray,
        thrust_ref: float,
        scenarios: List[ScenarioParams],
    ) -> Dict[str, np.ndarray | float]:
        """Solve one finite-horizon scenario-based SMPC problem."""
        nx = 4
        nu = 3
        N = self.cfg.horizon
        S = len(scenarios)

        opti = ca.Opti()

        # Shared control trajectory across scenarios, matching the paper's SMPC idea.
        U = opti.variable(nu, N)

        # One state trajectory per scenario.
        X = [opti.variable(nx, N + 1) for _ in range(S)]

        # Optional algebraic outputs for logging/inspection.
        T = [opti.variable(1, N) for _ in range(S)]

        # Initial conditions per scenario use the same state estimate.
        for i in range(S):
            opti.subject_to(X[i][:, 0] == x_est)

        # Bounds and move limits on shared input sequence.
        u_min = np.array(self.cfg.u_min, dtype=float)
        u_max = np.array(self.cfg.u_max, dtype=float)
        du_max = np.array(self.cfg.du_max, dtype=float)

        for k in range(N):
            opti.subject_to(opti.bounded(u_min[0], U[0, k], u_max[0]))
            opti.subject_to(opti.bounded(u_min[1], U[1, k], u_max[1]))
            opti.subject_to(opti.bounded(u_min[2], U[2, k], u_max[2]))

            if k == 0:
                opti.subject_to(ca.fabs(U[:, k] - u_prev) <= du_max)
            else:
                opti.subject_to(ca.fabs(U[:, k] - U[:, k - 1]) <= du_max)

        # Cost: expected sum across scenarios of tracking + current + control effort.
        J = 0
        for i, p in enumerate(scenarios):
            for k in range(N):
                xk = X[i][:, k]
                uk = U[:, k]
                xkp1, thrust_k, _nu_e, _mu, _eta = scenario_dynamics_symbolic(
                    xk, uk, p, self.nominal, self.cfg.dt
                )
                opti.subject_to(X[i][:, k + 1] == xkp1)
                opti.subject_to(T[i][0, k] == thrust_k)

                thrust_err = thrust_k - thrust_ref
                current_err = X[i][1, k + 1] - self.nominal.I_d_nom
                u_err = uk - np.array([self.nominal.mdot_nom, self.nominal.B_nom, self.nominal.V_d_nom])

                if k == 0:
                    du = uk - u_prev
                else:
                    du = uk - U[:, k - 1]

                # Mild state anchoring keeps the dummy model well-behaved.
                state_dev = X[i][:, k + 1] - np.array([
                    self.nominal.beta_bohm_nom * E_CHARGE * self.nominal.B_nom / M_E,
                    self.nominal.I_d_nom,
                    self.nominal.mdot_nom,
                    self.nominal.B_nom,
                ])

                J += (
                    self.cfg.w_thrust * thrust_err**2
                    + self.cfg.w_current * current_err**2
                    + self.cfg.w_input * ca.sumsqr(u_err)
                    + self.cfg.w_rate * ca.sumsqr(du)
                    + self.cfg.w_state * ca.sumsqr(state_dev)
                )

        J = J / max(S, 1)
        opti.minimize(J)

        # Warm start: hold previous input over the horizon.
        U_init = np.tile(u_prev.reshape(-1, 1), (1, N))
        opti.set_initial(U, U_init)
        for i in range(S):
            x_guess = np.tile(x_est.reshape(-1, 1), (1, N + 1))
            opti.set_initial(X[i], x_guess)

        opts = {
            "expand": True,
            "ipopt": {
                "print_level": 0,
                "max_iter": 400,
                "tol": 1e-5,
                "acceptable_tol": 1e-4,
                "sb": "yes",
            },
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            U_star = np.array(sol.value(U), dtype=float)
            X_star = [np.array(sol.value(Xi), dtype=float) for Xi in X]
            T_star = [np.array(sol.value(Ti), dtype=float).reshape(-1) for Ti in T]
            status = "success"
        except RuntimeError:
            # Safe fallback if the NLP fails.
            U_star = U_init
            X_star = [np.tile(x_est.reshape(-1, 1), (1, N + 1)) for _ in range(S)]
            T_star = [np.zeros(N, dtype=float) for _ in range(S)]
            status = "fallback"

        u0 = U_star[:, 0]
        thrust_pred_mean = float(np.mean([arr[0] for arr in T_star])) if T_star else float("nan")

        return {
            "status": status,
            "u_apply": u0,
            "U_star": U_star,
            "X_star": np.array(X_star),
            "T_star": np.array(T_star),
            "predicted_thrust_mean": thrust_pred_mean,
        }


# -----------------------------------------------------------------------------
# Closed-loop simulation helpers
# -----------------------------------------------------------------------------
def propagate_true_system(
    x: np.ndarray,
    u: np.ndarray,
    p_true: ScenarioParams,
    nominal: ThrusterNominal,
    dt: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Propagate one 'true' plant step using the same reduced-order surrogate."""
    x_sym = ca.DM(x)
    u_sym = ca.DM(u)
    x_next_sym, thrust_sym, nu_e_sym, mu_sym, eta_sym = scenario_dynamics_symbolic(
        x_sym, u_sym, p_true, nominal, dt
    )

    x_next = np.array(x_next_sym.full()).reshape(-1)
    outputs = {
        "thrust": float(thrust_sym),
        "nu_e_total": float(nu_e_sym),
        "mu_perp": float(mu_sym),
        "eta_acc": float(eta_sym),
        "I_d": float(x_next[1]),
        "nu_anom": float(x_next[0]),
        "mdot_a": float(x_next[2]),
        "B_peak": float(x_next[3]),
    }
    return x_next, outputs


def build_initial_state(nominal: ThrusterNominal) -> np.ndarray:
    """Create a dummy initial state estimate consistent with the surrogate."""
    nu_anom0 = nominal.beta_bohm_nom * (E_CHARGE * nominal.B_nom / M_E)
    return np.array([nu_anom0, nominal.I_d_nom, nominal.mdot_nom, nominal.B_nom], dtype=float)


def build_true_scenario(nominal: ThrusterNominal) -> ScenarioParams:
    """One dummy 'true plant' realization."""
    return ScenarioParams(
        beta_bohm=1.08 * nominal.beta_bohm_nom,
        nu_class=1.12 * nominal.nu_class_nom,
        n_e=0.95 * nominal.n_e_nom,
        eta_acc_bias=-0.03,
        Id_bias=0.18,
    )


def generate_thrust_reference(num_steps: int, nominal: ThrusterNominal) -> np.ndarray:
    """A simple thrust schedule to track."""
    ref = np.ones(num_steps) * nominal.thrust_nom
    if num_steps >= 10:
        ref[num_steps // 4 : num_steps // 2] = 0.095
        ref[num_steps // 2 : 3 * num_steps // 4] = 0.080
        ref[3 * num_steps // 4 :] = 0.090
    return ref


# -----------------------------------------------------------------------------
# Main example
# -----------------------------------------------------------------------------
def run_demo(num_steps: int = 40) -> Dict[str, np.ndarray]:
    nominal = ThrusterNominal()
    cfg = SMPCConfig()
    controller = HallThrusterSMPC(nominal, cfg)

    # Dummy estimated state values requested by the user.
    x_est = build_initial_state(nominal)
    u_prev = np.array([nominal.mdot_nom, nominal.B_nom, nominal.V_d_nom], dtype=float)

    # Dummy true plant realization and SMPC scenario bank.
    p_true = build_true_scenario(nominal)
    thrust_ref = generate_thrust_reference(num_steps, nominal)

    history = {
        "time": [],
        "thrust_ref": [],
        "thrust": [],
        "I_d": [],
        "nu_anom": [],
        "mdot_a": [],
        "B_peak": [],
        "nu_e_total": [],
        "mu_perp": [],
        "eta_acc": [],
        "mdot_cmd": [],
        "B_cmd": [],
        "Vd_cmd": [],
        "solve_status": [],
        "predicted_thrust_mean": [],
    }

    x_true = x_est.copy()

    for k in range(num_steps):
        # Resample scenarios each control update, like scenario-based SMPC.
        scenarios = sample_scenarios(nominal, cfg)
        sol = controller.solve(x_est=x_est, u_prev=u_prev, thrust_ref=float(thrust_ref[k]), scenarios=scenarios)
        u_apply = np.array(sol["u_apply"], dtype=float).reshape(-1)

        x_true, plant = propagate_true_system(x_true, u_apply, p_true, nominal, cfg.dt)

        # In this dummy example, assume the estimator gives exact state feedback.
        x_est = x_true.copy()
        u_prev = u_apply.copy()

        history["time"].append(k * cfg.dt)
        history["thrust_ref"].append(float(thrust_ref[k]))
        history["thrust"].append(plant["thrust"])
        history["I_d"].append(plant["I_d"])
        history["nu_anom"].append(plant["nu_anom"])
        history["mdot_a"].append(plant["mdot_a"])
        history["B_peak"].append(plant["B_peak"])
        history["nu_e_total"].append(plant["nu_e_total"])
        history["mu_perp"].append(plant["mu_perp"])
        history["eta_acc"].append(plant["eta_acc"])
        history["mdot_cmd"].append(float(u_apply[0]))
        history["B_cmd"].append(float(u_apply[1]))
        history["Vd_cmd"].append(float(u_apply[2]))
        history["solve_status"].append(str(sol["status"]))
        history["predicted_thrust_mean"].append(float(sol["predicted_thrust_mean"]))

    return {k: np.asarray(v) for k, v in history.items()}


def maybe_plot(history: Dict[str, np.ndarray]) -> None:
    """Plot results if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    t_ms = 1e3 * history["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_ms, history["thrust_ref"], label="Thrust reference")
    axes[0].plot(t_ms, history["thrust"], label="Thrust")
    axes[0].set_ylabel("N")
    axes[0].set_title("Thrust tracking")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_ms, history["mdot_cmd"] * 1e6, label="mdot command [mg/s]")
    axes[1].plot(t_ms, history["B_cmd"] * 1e3, label="B command [mT]")
    axes[1].plot(t_ms, history["Vd_cmd"], label="Vd command [V]")
    axes[1].set_ylabel("Command")
    axes[1].set_title("Control inputs")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t_ms, history["I_d"], label="Discharge current [A]")
    axes[2].plot(t_ms, history["eta_acc"], label="eta_acc [-]")
    axes[2].plot(t_ms, history["mu_perp"] * 1e3, label="mu_perp x1e3")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("State / output")
    axes[2].set_title("HallThruster-inspired internal quantities")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('hall_thruster_plot.png')
    print("Plot saved to hall_thruster_plot.png")


if __name__ == "__main__":
    hist = run_demo(num_steps=40)

    print("Final thrust      : {:.6f} N".format(hist["thrust"][-1]))
    print("Final thrust ref  : {:.6f} N".format(hist["thrust_ref"][-1]))
    print("Final I_d         : {:.4f} A".format(hist["I_d"][-1]))
    print("Final nu_anom     : {:.4e} 1/s".format(hist["nu_anom"][-1]))
    print("Final mdot_a      : {:.4e} kg/s".format(hist["mdot_a"][-1]))
    print("Final B_peak      : {:.4f} T".format(hist["B_peak"][-1]))
    print("Solver statuses   :", sorted(set(hist["solve_status"].tolist())))

    maybe_plot(hist)
