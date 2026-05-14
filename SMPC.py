"""
This module implements a continuous-time Cart-Pole simulation 
paired with a Stochastic Model Predictive Control (SMPC) controller
"""

# Simulation imports
import numpy as np
from scipy.integrate import solve_ivp

# Controller imports
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm
from casadi import *

# ==========================================
# CONSTANTS
# ==========================================
g = 9.81 # gravity (m/s^2)

@dataclass
class CartPoleNominal:
    m_c_nom: float  = 1.0    # Cart Mass (kg)
    m_p_nom: float  = 0.1    # Pole Mass (kg)
    l_nom: float    = 0.5      # Pole Length (m)


@dataclass
class ScenarioParams:
    m_c: float
    m_p: float
    l: float

# Take samples from gaussian distribution for each parameter to stochastisize the MPC
def sample_Scenarios(
    nominal: CartPoleNominal,
    num_samples: int,
    ParamSpread: Dict[str, float] | None = None
) -> List[ScenarioParams]:

    scenarios: List[ScenarioParams] = []

    if ParamSpread is None:
        # Default Parameters
        ParamSpread = {
            "cart_mass_mean":   1.0,      # nominal.m_c_com
            "cart_mass_var":    0.25,      
            "pole_mass_mean":   0.1,      # nominal.m_p_com
            "pole_mass_var":    0.025,
            "length_mean":      0.5,         # nominal.l_com
            "length_var":       0.0001        # maybe its a little stretchy
        }
    
    # Seperate Distributions
    # loc = mean, scale = stddev
    cart_dist =     norm(loc=ParamSpread["cart_mass_mean"], scale=ParamSpread["cart_mass_var"])
    pole_dist =     norm(loc=ParamSpread["pole_mass_mean"], scale=ParamSpread["pole_mass_var"])
    length_dist =   norm(loc=ParamSpread["length_mean"], scale=ParamSpread["length_var"])

    for i in range(num_samples):
        scenarios.append(
            ScenarioParams(
                m_c =   cart_dist.rvs(), 
                m_p =   pole_dist.rvs(),
                l =     length_dist.rvs()
            )
        )

    return scenarios



class CartPole:
    def __init__(self, m_c=1.0, m_p=0.1, l=0.5, g=9.81):
        """
        Initialize the Cart-Pole continuous-time simulation
        
        Parameters:
        m_c (float): Mass of the cart (kg)
        m_p (float): Mass of the pole (point mass at the end) (kg)
        l (float): Length of the pole (m)
        g (float): Gravity acceleration (m/s^2)
        """
        self.m_c =  m_c
        self.m_p =  m_p
        self.l =    l
        self.g =    g
        
        # State vector: [x, x_dot, theta, theta_dot] = [cart position, cart velocity, pole angle, pole angular velocity]
        # Convention: theta = 0 is perfectly upright (unstable equilibrium). 
        # Positive x is to the right, positive theta is clockwise.
        self.state = np.zeros(4)
        self.time = 0.0

    def reset(self, initial_state=None):
        # Resets the state of the cart-pole to the given initial state or to zeros
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros(4)
        self.time = 0.0
        return self.state

    def _dynamics(self, t, state, u):
        """
        The non-linear equations of motion for the cart-pole system.
        Designed to be called by scipy's ODE solver.
        Parameters:
        t = time (not used in this system since it's time-invariant, but required by solve_ivp
        state = current state vector [x, x_dot, theta, theta_dot]
        u = control force 
        """
        x, x_dot, theta, theta_dot = state
        
        # Pre-compute trigonometric values
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Total mass
        total_m = self.m_c + self.m_p
        
        # Physics Model:
        # Calculate angular acceleration of the pole (theta_dot_dot)
        # Using the standard point-mass cart-pole derivation
        temp = (u + self.m_p * self.l * theta_dot**2 * sin_theta) / total_m
        theta_dot_dot = (self.g * sin_theta - cos_theta * temp) / (self.l * (1.0 - (self.m_p * cos_theta**2) / total_m))
            
        # Calculate linear acceleration of the cart (x_dot_dot)
        x_dot_dot = temp - (self.m_p * self.l * theta_dot_dot * cos_theta) / total_m
        
        # Return the derivatives of the state: [x_dot, x_dot_dot, theta_dot, theta_dot_dot]
        return [x_dot, x_dot_dot, theta_dot, theta_dot_dot]



    def step(self, u, dt):
        """
        Steps the simulation forward by `dt` seconds applying control force `u`.
        
        Parameters:
        u (float): The control force applied to the cart in Newtons.
        dt (float): The duration of the control step in seconds.
        
        Returns:
        state (np.ndarray): The new state of the system after dt.
        """
        # We use solve_ivp to accurately integrate the continuous-time dynamics
        # over the interval [0, dt], holding the control `u` constant.
        res = solve_ivp(
            fun=lambda t, y: self._dynamics(t, y, u),
            t_span=(0, dt),
            y0=self.state,
            method='RK45' # Runge-Kutta 4(5) continuous time solver
        )
        
        # Update internal state with the results at the end of the integration window
        self.state = res.y[:, -1]
        self.time += dt
        
        return self.state







class SMPC_Controller:
    def __init__(self, S: int, N: int, dt: float):
        """
        Initializes the SMPC Controller object.
        
        Parameters:
        S (int): Number of sampled scenarios
        N (int): Number of horizon steps
        dt (float): time step (s)
        """
        self.S = S
        self.N = N
        self.dt = dt

        self.x = vertcat(   MX.sym("x"), 
                            MX.sym("x_dot"),
                            MX.sym("theta"),
                            MX.sym("theta_dot"))
        
        self.u = MX.sym("F")

        self.A_mat = vertcat(
                horzcat(0, 1, 0, 0),
                horzcat(0, 0, m*g/M, 0),
                horzcat(0, 0, 0, 1),
                horzcat(0, 0, g*(M + m)/(M*l), 0),
        )


        

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Initialize the system
    env = CartPole()
    
    # 2. Reset to a specific state: [x=0, x_dot=0, theta=0.1 rad, theta_dot=0]
    # (Slightly tilted off the unstable upright equilibrium)
    current_state = env.reset([0.0, 0.0, 0.1, 0.0])
    
    # 3. Simulate a control loop running at 50 Hz (dt = 0.02s)
    dt = 0.02
    simulation_duration = 2.0 # Simulate 2 seconds total
    num_steps = int(simulation_duration / dt)
    
    print("Initial state:", current_state)
    
    for step_idx in range(num_steps):
        # --- Control Algorithm Placeholder ---
        # Here is where you would calculate `u` based on the current_state.
        # For this example, let's just apply 0 force to watch the pole fall.
        u = 0.0 
        
        # Step the continuous-time physics forward
        current_state = env.step(u, dt)
        
    print(f"State after {simulation_duration}s of 0 force:", current_state)