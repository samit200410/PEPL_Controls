"""
This module implements a continuous-time Cart-Pole simulation 
paired with a Stochastic Model Predictive Control (SMPC) controller

A cart-pole is randomly generated at the start of each run. The controller
DOES NOT know the exact true mass/length. At every control step, it samples many
possible cart-pole parameter sets, predicts the future for each one, scores many
candidate force sequences, and applies only the first force from the best sequence.
"""

from __future__ import annotations

import argparse
from html import parser
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# Import CasADi
try:
    import casadi as ca
except ImportError as exc:
    raise SystemExit("Install CasADi to run this script") from exc

# ==========================================
# CONSTANTS & DATA CLASSES
# ==========================================

G = 9.81                    # gravity (m/s^2)

@dataclass
class CartPoleNominal:
    m_c_nom: float  = 5.0   # Nominal Cart Mass (kg)
    m_p_nom: float  = 1.0   # Nominal Pole Mass (kg)
    l_nom: float    = 1.0   # Nominal Pole Length (m)


@dataclass
class ScenarioParams:
    m_c: float              # Cart Mass (kg)
    m_p: float              # Pole Mass (kg)
    l: float                # Pole Length (m)

@dataclass
class SMPCConfig:
    
    # Timing & Horizon
    dt: float                       = 0.1                  # Period (time-step) for control updates (s)
    horizon_steps: int              = 100                    # Number of steps in the MPC horizon
    num_scenarios: int              = 15                    # Number of sampled scenarios to consider
    prediction_substeps: int        = 1                     # Number of sub-steps for prediction within each control step (accuracy)
    seed: int | None = None                                 # Random seed for reproducibility

    # Actuator Limits
    force_limit: float              = 15.0                  # Max magnitude of control force (N)
    delta_force_limit: float        = 17.5                  # Max change in control force per time step (N/s)
    
    # Physical/Safety Limits
    rail_limit: float               = 5.0                   # Max horizontal movement of the cart from center (m)
    fail_angle_rad: float           = math.radians(60.0)    # Max pole angle from vertical before failure (radians)
    rail_warning_fraction: float    = 0.75                  # Fraction of rail limit to start penalizing in cost function
    angle_warning_fraction: float   = 0.25                  # Pole angle to start penalizing in cost function (radians)
    
    # Cost Weights --> Greater the number, the more the controller cares
    q_x: float                      = 50.0                  # Cart position error 
    q_x_dot: float                  = 8.0                   # Cart velocity error 
    q_theta: float                  = 2500.0                # Pole angle error
    q_theta_dot: float             = 180.0                 # Pole angular velocity

    r_force: float                  = 0.01                  # Control effort (force magnitude)
    r_delta_force: float            = 0.01                  # Change in control effort (force smoothness)
    
    terminal_multiplier: float      = 15.0                  # Multiplier for the terminal cost (final state error)

    rail_warning_weight: float      = 5.0e2                 # Additional cost weight when cart is near rail limit
    rail_limit_weight: float        = 5.0e7                 # Additional cost weight when cart exceeds rail limit
    angle_warning_weight: float     = 5.0e4                 # Additional cost weight when pole is near fail angle
    angle_limit_weight: float       = 5.0e7                 # Additional cost weight when pole exceeds fail angle

    # Solver Settings
    ipopt_max_iter: int             = 50                    # Maximum iterations for IPOPT solver
    ipopt_tol: float                = 1e-3                  # Tolerance for IPOPT solver convergence
    ipopt_acceptable_tol: float     = 1e-2                  # Acceptable tolerance for IPOPT solver convergence
    ipopt_print_level: int          = 0                    # Number of iterations to meet acceptable tolerance before stopping

    # Uncertainty used when Sampling Scenarios 
    cart_mass_uncertainty: float    = 0.05                   # Standard deviation for cart mass sampling (fraction of nominal)
    pole_mass_uncertainty: float    = 0.05                  # Standard deviation for pole mass sampling (fraction of nominal)
    pole_length_uncertainty: float  = 0.05                  # Standard deviation for pole length sampling (fraction of nominal)


# ==========================================
# Cart-Pole System Physics
# ==========================================

def _positive_sample(value: float, minimum: float) -> float:
    return max(float(value), minimum)

def sample_scenarios(
        nominal: CartPoleNominal,
        cfg: SMPCConfig,
        rng: np.random.Generator,
        num_samples: int | None = None,
) -> List[ScenarioParams]:
    
    if num_samples is None:
        num_samples = cfg.num_scenarios

    # Define the spread (standard deviation) for each parameter's distribution
    scenarios: List[ScenarioParams] = []
    
    for _ in range(num_samples):
        m_c = _positive_sample(rng.normal(nominal.m_c_nom, cfg.cart_mass_uncertainty * nominal.m_c_nom), minimum=0.25)
        m_p = _positive_sample(rng.normal(nominal.m_p_nom, cfg.pole_mass_uncertainty * nominal.m_p_nom), minimum=0.05)
        l = _positive_sample(rng.normal(nominal.l_nom, cfg.pole_length_uncertainty * nominal.l_nom), minimum=0.1)
        scenarios.append(ScenarioParams(m_c=m_c, m_p=m_p, l=l))

    return scenarios

# Create a set of scenarios for the controller to explicitly consider
def build_controller_scenarios(
        nominal: CartPoleNominal,
        cfg: SMPCConfig,
        rng: np.random.Generator
) -> List[ScenarioParams]:
    
    scenarios: List[ScenarioParams] = [
        # Nominal - Best Guess
        ScenarioParams(nominal.m_c_nom, nominal.m_p_nom, nominal.l_nom),
        
        # Extreme corners of the uncertainty box (min and max for each parameter) --> 95% confidence interval if we assume normal distribution and 2 std devs
        # Minimums
        ScenarioParams(
            _positive_sample(nominal.m_c_nom * (1.0 - 2.0 * cfg.cart_mass_uncertainty), 0.25),
            _positive_sample(nominal.m_p_nom * (1.0 - 2.0 * cfg.pole_mass_uncertainty), 0.05),
            _positive_sample(nominal.l_nom * (1.0 - 2.0 * cfg.pole_length_uncertainty), 0.1),
        ),
        # Maximums
        ScenarioParams(
            _positive_sample(nominal.m_c_nom * (1.0 + 2.0 * cfg.cart_mass_uncertainty), 0.25),
            _positive_sample(nominal.m_p_nom * (1.0 + 2.0 * cfg.pole_mass_uncertainty), 0.05),
            _positive_sample(nominal.l_nom * (1.0 + 2.0 * cfg.pole_length_uncertainty), 0.1),
        ),
        # Normal Cart, Fast Pole
        ScenarioParams(
            nominal.m_c_nom,
            _positive_sample(nominal.m_p_nom * (1.0 - 2.0 * cfg.pole_mass_uncertainty), 0.05),
            _positive_sample(nominal.l_nom * (1.0 - 2.0 * cfg.pole_length_uncertainty), 0.1),
        ),
    ]

    if cfg.num_scenarios <= len(scenarios):
        return scenarios[: cfg.num_scenarios]

    scenarios.extend(
        sample_scenarios(nominal, cfg, rng, num_samples=cfg.num_scenarios - len(scenarios))
    )
    return scenarios  

def cartpole_system_derivatives(
        state: np.ndarray,
        force: float,
        params: ScenarioParams
) -> np.ndarray:
    
    x, x_dot, theta, theta_dot = state
    sin_theta = np.sin(float(theta))
    cos_theta = np.cos(float(theta))

    m_c, m_p, l = params.m_c, params.m_p, params.l

    total_mass = m_c + m_p

    temp_calc = (force + m_p * l * theta_dot**2 * sin_theta) / total_mass
    
    theta_dot_dot = (G * sin_theta - cos_theta * temp_calc) / (l * (1.0 - (m_p * cos_theta**2) / total_mass))
    
    x_dot_dot = temp_calc - (m_p * l * theta_dot_dot * cos_theta) / total_mass

    return np.array([x_dot, x_dot_dot, theta_dot, theta_dot_dot], dtype=float)


# def rk6_step(
#         state: np.ndarray,
#         force: float,
#         dt: float,
#         params: ScenarioParams
# ) -> np.ndarray:
#     k1 = cartpole_system_derivatives(state, force, params)
#     k2 = cartpole_system_derivatives(state + 0.25 * dt * k1, force, params)
#     k3 = cartpole_system_derivatives(state + (3/32) * dt * k1 + (9/32) * dt * k2, force, params)
#     k4 = cartpole_system_derivatives(state + (1932/2197) * dt * k1 - (7200/2197) * dt * k2 + (7296/2197) * dt * k3, force, params)
#     k5 = cartpole_system_derivatives(state + (439/216) * dt * k1 - 8 * dt * k2 + (3680/513) * dt * k3 - (845/4104) * dt * k4, force, params)
#     k6 = cartpole_system_derivatives(state - (8/27) * dt * k1 + 2 * dt * k2 - (3544/2565) * dt * k3 + (1859/4104) * dt * k4 - (11/40) * dt * k5, force, params)
    
#     new_state = state + (16/135) * dt * k1 + (6656/12825) * dt * k3 + (28561/56430) * dt * k4 - (9/50) * dt * k5 + (2/55) * dt * k6
    
#     return new_state


def rk4_step(
      state: np.ndarray, 
      force: float,
      dt: float,
      params: ScenarioParams
) -> np.ndarray:
    k1 = cartpole_system_derivatives(state, force, params)
    k2 = cartpole_system_derivatives(state + 0.5 * dt * k1, force, params)
    k3 = cartpole_system_derivatives(state + 0.5 * dt * k2, force, params)
    k4 = cartpole_system_derivatives(state + dt * k3, force, params)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return new_state


class CartPolePlant:
    def __init__(self, params: ScenarioParams):
        self.params = params
        self.state = np.zeros(4, dtype=float)  # [x, x_dot, theta, theta_dot]
        self.time = 0.0

    def reset(self, initial_state: np.ndarray | List[float]) -> np.ndarray:
        self.state = np.array(initial_state, dtype=float)
        self.time = 0.0
        return self.state.copy()

    def step(self, force: float, dt: float) -> np.ndarray:
        self.state = rk4_step(self.state, force, dt, self.params)
        self.time += dt
        return self.state.copy()
    
# ==========================================
# Symbolic Prediction Plants for Controller
# ==========================================

def cartpole_symbolic_derivatives(
        x: ca.MX,
        u: ca.MX,
        m_c: ca.MX,
        m_p: ca.MX,
        l: ca.MX
) -> ca.MX:
    
    x_pos = x[0]
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    
    total_mass = m_c + m_p

    temp_calc = (u + m_p * l * theta_dot**2 * sin_theta) / total_mass

    theta_dot_dot = (G * sin_theta - cos_theta * temp_calc) / (l * (1.0 - (m_p * cos_theta**2) / total_mass))
    
    x_dot_dot = temp_calc - (m_p * l * theta_dot_dot * cos_theta) / total_mass

    return ca.MX(ca.vertcat(x_dot, x_dot_dot, theta_dot, theta_dot_dot))

# def rk6_symbolic_step(
#         x: ca.MX,
#         u: ca.MX,
#         dt: float,
#         substeps: int,
#         m_c: ca.MX,
#         m_p: ca.MX,
#         l: ca.MX
# ) -> ca.MX:
    
#     h = dt / max(substeps, 1)

#     x_next = x

#     for _ in range(max(substeps, 1)):
#         k1 = cartpole_symbolic_derivatives(x_next, u, m_c, m_p, l)
#         k2 = cartpole_symbolic_derivatives(x_next + 0.25 * h * k1, u, m_c, m_p, l)
#         k3 = cartpole_symbolic_derivatives(x_next + (3/32) * h * k1 + (9/32) * h * k2, u, m_c, m_p, l)
#         k4 = cartpole_symbolic_derivatives(x_next + (1932/2197) * h * k1 - (7200/2197) * h * k2 + (7296/2197) * h * k3, u, m_c, m_p, l)
#         k5 = cartpole_symbolic_derivatives(x_next + (439/216) * h * k1 - 8 * h * k2 + (3680/513) * h * k3 - (845/4104) * h * k4, u, m_c, m_p, l)
#         k6 = cartpole_symbolic_derivatives(x_next - (8/27) * h * k1 + 2 * h * k2 - (3544/2565) * h * k3 + (1859/4104) * h * k4 - (11/40) * h * k5, u, m_c, m_p, l)

#         x_next = x_next + (16/135) * h * k1 + (6656/12825) * h * k3 + (28561/56430) * h * k4 - (9/50) * h * k5 + (2/55) * h * k6

#     return x_next


def  _symbolic_step(
    x: ca.MX,
    u: ca.MX,
    dt: float,
    substeps: int,
    m_c: ca.MX,
    m_p: ca.MX,
    l: ca.MX,
) -> ca.MX:

    h = dt / max(substeps, 1)
    
    x_next = x

    for _ in range(max(substeps, 1)):
        k1 = cartpole_symbolic_derivatives(x_next, u, m_c, m_p, l)
        k2 = cartpole_symbolic_derivatives(x_next + 0.5 * h * k1, u, m_c, m_p, l)
        k3 = cartpole_symbolic_derivatives(x_next + 0.5 * h * k2, u, m_c, m_p, l)
        k4 = cartpole_symbolic_derivatives(x_next + h * k3, u, m_c, m_p, l)
        x_next = x_next + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_next


# ==========================================
# CasADi-based SMPC Controller
# ==========================================

class SMPC_Controller:
    def __init__(self, nominal:CartPoleNominal, cfg: SMPCConfig):
        self.nominal = nominal
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.previous_force = 0.0
        self.previous_solution: np.ndarray | None = None

        # Fixed Scenario Set for the Controller to Consider
        # IMPORTANT: For HET implementation, values can be replaced with posterior samples updated at each time step instead of fixed scenarios
        self.scenarios = build_controller_scenarios(nominal, cfg, self.rng)

        self._build_solver()


    def _build_solver(self) -> None:
        cfg = self.cfg
        N = cfg.horizon_steps
        S = cfg.num_scenarios
        nx = 4                      # Number of State Variables (x, x_dot, theta, theta_dot)

        optimizer = ca.Opti()
        self.optimizer = optimizer

        # Updated Parameters for each scenario
        self.x0_param = optimizer.parameter(nx, 1)              # Initial state parameter
        self.u_previous_param = optimizer.parameter(1, 1)       # Previous control input (force)
        self.scenario_param = optimizer.parameter(3, S)         # Scenario parameters (m_c, m_p, l for each scenario)

        # Decision Variables
        self.X = [optimizer.variable(nx, N + 1) for _ in range(S)]      # State trajectory for each scenario
        self.U = optimizer.variable(1, N)                               # Control trajectory

        U = self.U
        X = self.X

        # Force and Force Change Constraints
        for k in range(N):
            
            # Absolute Limit on Force
            optimizer.subject_to(optimizer.bounded(-cfg.force_limit, U[0, k], cfg.force_limit))

            if k == 0:
                delta_u = U[0, k] - self.u_previous_param[0, 0]
            else:
                delta_u = U[0, k] - U[0, k - 1]

            optimizer.subject_to(
                optimizer.bounded(
                    -cfg.delta_force_limit,
                    delta_u,
                    cfg.delta_force_limit,
                )
            )

        J = 0

        rail_warning_threshold = cfg.rail_warning_fraction * cfg.rail_limit
        angle_warning_threshold = cfg.angle_warning_fraction * cfg.fail_angle_rad

        for s in range(S):
            m_c_s = self.scenario_param[0, s]
            m_p_s = self.scenario_param[1, s]
            l_s = self.scenario_param[2, s]

            # Initial Condition Constraint
            optimizer.subject_to(X[s][:, 0] == self.x0_param)

            for k in range(N):
                x_k = X[s][:, k]
                u_k = U[0, k]
                
                x_next_prediction = rk4_symbolic_step(x_k, u_k, cfg.dt, cfg.prediction_substeps, m_c_s, m_p_s, l_s)
                
                optimizer.subject_to(X[s][:, k + 1] == x_next_prediction)

                x_pos = X[s][0, k+1]
                x_vel = X[s][1, k+1]
                theta = X[s][2, k+1]
                theta_dot = X[s][3, k+1]

                if k == 0:
                    delta_u = u_k - self.u_previous_param[0, 0]
                else:
                    delta_u = u_k - U[0, k - 1]



                # COST FUNCTION
                state_cost = (
                    cfg.q_x * x_pos**2 +
                    cfg.q_x_dot * x_vel**2 +
                    cfg.q_theta * theta**2 +
                    cfg.q_theta_dot * theta_dot**2
                )

                input_cost = cfg.r_force * u_k**2 + cfg.r_delta_force * delta_u**2

                # Additional "Close Call" Costs for Being Near or Beyond Limits
                rail_warning_cost = ca.fmax(ca.fabs(x_pos) - rail_warning_threshold, 0)
                rail_limit_cost = ca.fmax(ca.fabs(x_pos) - cfg.rail_limit, 0)
                angle_warning_cost = ca.fmax(ca.fabs(theta) - angle_warning_threshold, 0)
                angle_limit_cost = ca.fmax(ca.fabs(theta) - cfg.fail_angle_rad, 0)

                safety_cost = (
                    cfg.rail_warning_weight * rail_warning_cost**2 +
                    cfg.rail_limit_weight * rail_limit_cost**2 +
                    cfg.angle_warning_weight * angle_warning_cost**2 +
                    cfg.angle_limit_weight * angle_limit_cost**2
                )

                J += state_cost + input_cost + safety_cost

            # Terminal Cost
            x_terminal = X[s][:, N]
            terminal_cost = cfg.terminal_multiplier * (
                cfg.q_x * x_terminal[0] ** 2 + 
                cfg.q_x_dot * x_terminal[1] ** 2 +
                cfg.q_theta * x_terminal[2] ** 2 +
                cfg.q_theta_dot * x_terminal[3] ** 2
            )

            J += terminal_cost


        self.J = J / max(S,1)  # Average cost across scenarios
        optimizer.minimize(self.J)


        # Solver Options
        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "print_level": cfg.ipopt_print_level,
                "max_iter": cfg.ipopt_max_iter,
                "tol": cfg.ipopt_tol,
                "acceptable_tol": cfg.ipopt_acceptable_tol,
                "sb": "yes",
                "warm_start_init_point": "yes",
        
                # --- ADDITIONS FOR REAL-TIME MPC ---
                # "linear_solver": "mumps",                   # Or "mumps" / "ma27" / "ma57"
                # "hessian_approximation": "limited-memory", 
                # "max_cpu_time": cfg.max_cpu_time,         # Hard real-time limit
                # "mu_strategy": "adaptive",                  # Handle abrupt constraint changes
            },
        }

        optimizer.solver("ipopt", options)

        self._set_scenario_parameters(self.scenarios)

    def _set_scenario_parameters(self, scenarios: List[ScenarioParams]) -> None:
        scenario_values = np.array([[s.m_c, s.m_p, s.l] for s in scenarios], dtype=float).T
        self.optimizer.set_value(self.scenario_param, scenario_values)

    def _shift_previous_solution(self) -> np.ndarray:
        N = self.cfg.horizon_steps

        if self.previous_solution is None or len(self.previous_solution) != N:
                return np.zeros(N, dtype=float)
            
        # Shift the previous solution to use as an initial guess for the next optimization
        shifted_solution = np.empty(N, dtype=float)
        shifted_solution[:-1] = self.previous_solution[1:]  # Shift states forward
        shifted_solution[-1] = self.previous_solution[-1]    # Last state repeated
        return shifted_solution
        
    def _make_state_guess(
            self, current_state: np.ndarray, 
            force_guess: np.ndarray, 
            params: ScenarioParams
    ) -> np.ndarray:
        
        nx = 4
        N = self.cfg.horizon_steps
        X_guess = np.zeros((nx, N+1), dtype=float)
        X_guess[:, 0] = current_state
        x = current_state.copy()
         
        for k in range(N):
            x = rk4_step(x, float(force_guess[k]), self.cfg.dt, params)
            X_guess[:, k+1] = x
        return X_guess


    def solve(self, current_state: np.ndarray) -> Dict[str, object]:
        cfg = self.cfg
        U_initial = self._shift_previous_solution()

        self.optimizer.set_value(self.x0_param, current_state.reshape(-1, 1))
        self.optimizer.set_value(self.u_previous_param, np.array([[self.previous_force]], dtype=float))
        self.optimizer.set_initial(self.U, U_initial.reshape(1, -1))

        for s, params in enumerate(self.scenarios):
            self.optimizer.set_initial(self.X[s], self._make_state_guess(current_state, U_initial, params))

        try:
            solution = self.optimizer.solve()
            U_important = np.array(solution.value(self.U), dtype=float).reshape(-1)
            best_cost = float(solution.value(self.J))
            status = "success"

        except RuntimeError:
            
        # Preserve a safe plan if IPOPT fails
            U_important = U_initial.copy()
                
            try:
                best_cost = float(self.optimizer.debug.value(self.J))
            
            except Exception:
                best_cost = math.inf
            
            status = "fallback"

        force_to_apply = float(np.clip(U_important[0], -cfg.force_limit, cfg.force_limit))
        self.previous_force = force_to_apply
        self.previous_solution = U_important.copy()

        return {
            "force": force_to_apply,
            "best_cost": best_cost,
            "status": status,
            "U_important": U_important.copy(),
            "scenario_mean": scenario_summary(self.scenarios),
        }

# ==========================================
# Simulation Loop and Result Handling
# ==========================================

def scenario_summary(scenarios: List[ScenarioParams]) -> Tuple[float, float, float]:
    summary = np.array([[s.m_c, s.m_p, s.l] for s in scenarios], dtype=float)
    return tuple(np.mean(summary, axis=0))

def build_random_cartpole(nominal: CartPoleNominal,
                               cfg: SMPCConfig,
                               rng: np.random.Generator
) -> ScenarioParams:
    
    return sample_scenarios(nominal, cfg, rng, num_samples=1)[0]


def print_terminal_header() -> None:
    print(
        "\n"
        " step | time [s] | theta [deg] |   x[m]   | force[N] |   cost   | status\n"
        "------+----------+-------------+----------+----------+----------+---------"
    )


def print_progress_in_terminal(
        k: int,
        t: float,
        state: np.ndarray,
        force: float,
        cost: float,
        status: str
) -> None:
    
    theta_deg = math.degrees(float(state[2]))
    cost_str = "infinite" if not math.isfinite(cost) else f"{cost:8.2f}"
    
    print(
        f"{k:5d} | {t:8.3f} | {theta_deg:11.3f} | "
        f"{state[0]:8.3f} | {force:8.3f} | {cost_str:>8} | {status:>7}"
    )


def run_closed_loop_simulation(
    cfg: SMPCConfig,
    duration: float,
    initial_angle_deg: float,
    print_every: int,
) -> Dict[str, object]:
    
    nominal = CartPoleNominal()

    plant_rng = np.random.default_rng(None if cfg.seed is None else cfg.seed + 999)
    
    true_parameters = build_random_cartpole(nominal, cfg, plant_rng)

    plant = CartPolePlant(true_parameters)

    controller = SMPC_Controller(nominal, cfg)


    initial_state = np.array([0.0, 0.0, math.radians(initial_angle_deg), 0.0], dtype=float)
    state = plant.reset(initial_state)
    num_steps = int(duration / cfg.dt)

    history: Dict[str, List[float | str]] = {
        "time": [],
        "x": [],
        "x_dot": [],
        "theta": [],
        "theta_dot": [],
        "theta_deg": [],
        "force": [],
        "best_cost": [],
        "scenario_m_c_mean": [],
        "scenario_m_p_mean": [],
        "scenario_l_mean": [],
        "solve_status": [],
    }

    print("\nTrue Plant Values:")
    print(f"  cart mass m_c = {true_parameters.m_c:.4f} kg")
    print(f"  pole mass m_p = {true_parameters.m_p:.4f} kg")
    print(f"  pole length l = {true_parameters.l:.4f} m")
    
    print_terminal_header()

    failed = False
    fail_reason = ""

    for k in range(num_steps):
        solution = controller.solve(state)
        force = float(solution["force"])
        state = plant.step(force, cfg.dt)
        t = (k + 1) * cfg.dt

        m_c_mean, m_p_mean, l_mean = solution["scenario_mean"]

        history["time"].append(t)
        history["x"].append(float(state[0]))
        history["x_dot"].append(float(state[1]))
        history["theta"].append(float(state[2]))
        history["theta_dot"].append(float(state[3]))
        history["theta_deg"].append(math.degrees(float(state[2])))
        history["force"].append(force)
        history["best_cost"].append(float(solution["best_cost"]))
        history["scenario_m_c_mean"].append(float(m_c_mean))
        history["scenario_m_p_mean"].append(float(m_p_mean))
        history["scenario_l_mean"].append(float(l_mean))
        history["solve_status"].append(str(solution["status"]))

        if k % print_every == 0 or k == num_steps - 1:
            print_progress_in_terminal(k, t, state, force, float(solution["best_cost"]), str(solution["status"]))

        if abs(state[2]) > cfg.fail_angle_rad:
            failed = True
            fail_reason = "pole angle exceeded failure limit"
            break

        if abs(state[0]) > 1.5 * cfg.rail_limit:
            failed = True
            fail_reason = "cart moved too far from center"
            break

    numeric_keys = [
        "time",
        "x",
        "x_dot",
        "theta",
        "theta_dot",
        "theta_deg",
        "force",
        "best_cost",
        "scenario_m_c_mean",
        "scenario_m_p_mean",
        "scenario_l_mean",
    ]
    np_history: Dict[str, object] = {key: np.asarray(history[key], dtype=float) for key in numeric_keys}
    np_history["solve_status"] = np.asarray(history["solve_status"], dtype=object)

    return {
        "history": np_history,
        "true_params": true_parameters,
        "nominal": nominal,
        "cfg": cfg,
        "failed": failed,
        "fail_reason": fail_reason,
    }

def plot_history(results: Dict[str, object], output_path: Path) -> None:
    history = results["history"]
    true_params = results["true_params"]
    cfg = results["cfg"]
    t = history["time"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t, history["theta_deg"], label="pole angle")
    axes[0].axhline(math.degrees(cfg.fail_angle_rad), linestyle="--", label="angle limit")
    axes[0].axhline(-math.degrees(cfg.fail_angle_rad), linestyle="--")
    axes[0].set_ylabel("theta [deg]")
    axes[0].set_title("Balanced CasADi scenario SMPC cart-pole debug output")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, history["x"], label="cart position")
    axes[1].axhline(cfg.rail_limit, linestyle="--", label="rail limit")
    axes[1].axhline(-cfg.rail_limit, linestyle="--")
    axes[1].set_ylabel("x [m]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t, history["force"], label="applied force")
    axes[2].axhline(cfg.force_limit, linestyle="--", label="force limit")
    axes[2].axhline(-cfg.force_limit, linestyle="--")
    axes[2].set_ylabel("force [N]")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    axes[3].plot(t, history["best_cost"], label="optimized expected cost")
    axes[3].set_ylabel("cost")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="best")
    axes[3].grid(True)

    summary = (
        f"True plant: m_c={true_params.m_c:.3f} kg, "
        f"m_p={true_params.m_p:.3f} kg, l={true_params.l:.3f} m\n"
        f"SMPC: S={cfg.num_scenarios}, horizon={cfg.horizon_steps}, "
        f"dt={cfg.dt:.3f}s, force limit={cfg.force_limit:.1f}N, qx={cfg.q_x:.1f}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def summarize_results(results: Dict[str, object], output_path: Path | None) -> None:
    history = results["history"]
    failed = bool(results["failed"])
    fail_reason = str(results["fail_reason"])

    if len(history["time"]) == 0:
        print("No simulation history was produced.")
        return

    final_theta = history["theta_deg"][-1]
    final_x = history["x"][-1]
    max_theta = np.max(np.abs(history["theta_deg"]))
    max_x = np.max(np.abs(history["x"]))
    statuses = sorted(set(history["solve_status"].tolist()))

    print("\nFinal summary:")
    print(f"  status              : {'FAILED' if failed else 'completed'}")
    if failed:
        print(f"  failure reason      : {fail_reason}")
    print(f"  solver statuses     : {statuses}")
    print(f"  final pole angle    : {final_theta:.3f} deg")
    print(f"  max |pole angle|    : {max_theta:.3f} deg")
    print(f"  final cart position : {final_x:.3f} m")
    print(f"  max |cart position| : {max_x:.3f} m")
    if output_path is not None:
        print(f"  plot saved to       : {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced CasADi scenario SMPC for cart-pole.")

    # Simulation settings
    parser.add_argument("--duration", type=float, default=8.0, help="simulation time [s]")
    parser.add_argument("--initial-angle-deg", type=float, default=8.0, help="initial pole angle [deg]")

    # MPC timing / size
    parser.add_argument("--dt", type=float, default=0.02, help="control time step [s]")
    parser.add_argument("--horizon", type=int, default=40, help="prediction horizon steps")
    parser.add_argument("--scenarios", type=int, default=3, help="number of SMPC scenarios")
    parser.add_argument("--prediction-substeps", type=int, default=1, help="symbolic RK substeps per control step")

    # Actuator limits
    parser.add_argument("--force-limit", type=float, default=25.0, help="force limit [N]")
    parser.add_argument("--delta-force-limit", type=float, default=25.0, help="max force change per step [N]")

    # Random seed
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    # Cost weights
    parser.add_argument("--qx", type=float, default=300.0, help="cart position cost weight")
    parser.add_argument("--qx-dot", type=float, default=50.0, help="cart velocity cost weight")
    parser.add_argument("--qtheta", type=float, default=2500.0, help="pole angle cost weight")
    parser.add_argument("--qtheta-dot", type=float, default=220.0, help="pole angular velocity cost weight")
    parser.add_argument("--terminal", type=float, default=80.0, help="terminal cost multiplier")

    # Solver / output
    parser.add_argument("--max-iter", type=int, default=80, help="IPOPT max iterations")
    parser.add_argument("--print-every", type=int, default=25, help="print every N steps")
    parser.add_argument("--plot-file", type=str, default="SMPC_CpP.png")
    parser.add_argument("--no-plot", action="store_true", help="disable plotting")

    parser.add_argument("--r-force", type=float, default=0.05, help="control effort cost weight")
    parser.add_argument("--r-delta-force", type=float, default=1.0, help="control force smoothness cost weight")    

    return parser.parse_args()
        

# ==========================================
# Main Simulation Loop
# ==========================================

def main() -> None:
    args = parse_args()
    cfg = SMPCConfig(
        dt=args.dt,
        horizon_steps=args.horizon,
        num_scenarios=args.scenarios,
        prediction_substeps=args.prediction_substeps,
        force_limit=args.force_limit,
        delta_force_limit=args.delta_force_limit,
        seed=args.seed,
        q_x=args.qx,
        q_x_dot=args.qx_dot,
        q_theta=args.qtheta,
        q_theta_dot=args.qtheta_dot,
        r_force=args.r_force,
        r_delta_force=args.r_delta_force,
        terminal_multiplier=args.terminal,
        ipopt_max_iter=args.max_iter,
    )

    start_time = time.perf_counter()
    results = run_closed_loop_simulation(
        cfg=cfg,
        duration=args.duration,
        initial_angle_deg=args.initial_angle_deg,
        print_every=max(args.print_every, 1),
    )

    runtime = time.perf_counter() - start_time

    output_path = None
    if not args.no_plot:
        output_path = Path(args.plot_file)
        plot_history(results, output_path)

    summarize_results(results, output_path)
    print(f"  wall-clock runtime  : {runtime:.2f} s")


if __name__ == "__main__":
    main()
