import numpy as np
import matplotlib.pyplot as plt

# Controller
def PID_discharge_current(measured_current, desired_current, nominal_flow,
                          integral_error, previous_error, dt):
    
    # measured_current: Actual value being regulated
    #   Origin: DAQ or Diffusion Model Estimate
    # desired_current: Setpoint
    #   Origin: Operator set value
    # nominal_flow: Baseline flow rate --> Correctional Value
    #   Origin: Known value from thruster
    # integral_error: Accumulated past error
    #   Origin: Tuned internal controller memory
    # previous_error: error from last time step
    #   Origin: Tuned internal controller memory
    # dt
    #   Origin: Time-step between controller updates

    # PID Variable Gains 
    # TODO: Make these global if they won't change 
    Kp = None #FILL IN
    Ki = None #FILL IN
    Kd = None #FILL IN

    # Flow Safety
    flow_min = None
    flow_max = None
    
    # Integral Windup Prevention
    integral_min = None
    integral_max = None

    error = desired_current - measured_current

    integral_error += error * dt
    integral_error = max(min(integral_error, integral_max), integral_min)

    derivative_error = (error - previous_error) / dt
    
    control = Kp * error + Ki * integral_error + Kd * derivative_error

    flow_control = nominal_flow + control
    flow_control = max(min(flow_control, flow_max), flow_min)

    previous_error = error

    return flow_control, integral_error, previous_error, error



# Fake Thruster
def simulation_PID_discharge_current(current, flow, dt):

    K_system = 1.5
    tau = 0.8

    target_curret = K_system * flow

    next_current = current + (dt/tau) * target_current - current


## Time step should be run every 5 seconds or so
