from PID import PID_discharge_current, PID_magnetic_coil_current

import time

desired_current = 5.0
nominal_flow = 5.0
dt = 0.01

integral_error = 0.0
previous_error = 0.0

for i in range(10):
    measured_current = 4.2 + 0.05 * i 

    flow_command, integral_error, previous_error, error = PID_discharge_current(
        measured_current,
        desired_current,
        nominal_flow,
        integral_error,
        previous_error,
        dt
    )

    print("Measured current:", measured_current)
    print("Error:", error)
    print("Flow command:", flow_command)
    print("-----")

    time.sleep(dt)
