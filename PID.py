import numpy as np
import sys, time, struct, serial
import multiprocessing as mp
import socket

# ----------------------------
# Safety Limites

VOLTAGE_MIN = 200.0
VOLTAGE_MAX = 300.0
POWER_TARGET = 2000.0

INNER_MAG_MIN = 3.5
INNER_MAG_MAX = 6.0

OUTER_MAG_MIN = 4.0
OUTER_MAG_MAX = 7.5

ANODE_FLOW_MIN = 60.0
ANODE_FLOW_MAX = 120.0

CATHODE_FRAC_MIN = 0.05
CATHODE_FRAC_NOMINAL = 0.07
CATHODE_FRAC_MAX = 0.10
# ----------------------------

# ----------------------------
# PID Turning

# Mass Flow
FLOW_KP = 2.0
FLOW_KI = 0.2
FLOW_KD = 0.0

# Magnetic Field
MAG_KP = None
MAG_KI = None
MAG_KD = None
# ----------------------------




# TCP Server Config (TX)
HOST = socket.gethostname(socket.gethostbyname())  
TCP_PORT_RX = 6700 
TCP_RX = (HOST, TCP_PORT_RX)

# TCP Client Config (RX)
LABVIEW_IP = '10.0.0.1'
TCP_PORT_TX = 6701 # TODO: Set this to the actual TCP port for receiving data from LabView
TCP_TX = (LABVIEW_IP, TCP_PORT_TX)

HEADER = 4
COMMAND = 1  
LENGTH = 4

# Shared Data
latest_packet = None

# Locks for shared data
reader_lock = mp.Lock()
writer_lock = mp.Lock()

def TCP_server_thread(conn, addr):
    with conn:
        print('Connected by', addr)
        while True:
            header = conn.recv(HEADER)
            if not header: break

            arr_size = struct.unpack('!I', header)[0]  # Network byte order (big-endian)


            cmd = conn.recv(COMMAND)
            if not cmd: break
            msg_cmd = cmd.decode('utf-8').strip()
            dat_len = conn.recv(LENGTH)
            if not dat_len: break
            msg_len = dat_len.decode('utf-8').strip()
            data = conn.recv(int(msg_len))
            if not data: break
            print(f"Received command: {msg_cmd}, data length: {msg_len}, data: {data}")


def PID_threadspawner():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(TCP_TX)

    # TODO: Implement thread spawner for PID control of two processes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(TCP_RX)
        s.listen(1)
        conn, addr = s.accept()

    # Start thread for packet send and receive
    reader_thread = mp.Process(target=TCP_server_thread, args=(conn, addr))
    reader_thread.start()
    
    # Start thread for discharge current controls
    discharge_current_thread = mp.Process(target=PID_discharge_current, args=(...)) # TODO: Fill in arguments
    discharge_current_thread.start()

    # Start thread for magnetic coil current control
    magnetic_coil_thread = mp.Process(target=PID_magnetic_coil_current, args=(...)) # TODO: Fill in arguments
    magnetic_coil_thread.start()

    reader_thread.join()
    discharge_current_thread.join()
    magnetic_coil_thread.join()



def PID_discharge_current(measured_current, desired_current, nominal_flow,
                          integral_error, previous_error, dt):
    
    # TODO: Implement Testbench and tune variables

    # PID Variable Gains 
    # TODO: Make these global if they won't change 
    Kp = FLOW_KP
    Ki = FLOW_KI
    Kd = FLOW_KD

    # Flow Safety
    flow_min = ANODE_FLOW_MIN
    flow_max = ANODE_FLOW_MAX
    
    # Integral Windup Prevention
    integral_min = -20.0
    integral_max = 20.0

    error = desired_current - measured_current

    integral_error += error * dt
    integral_error = max(min(integral_error, integral_max), integral_min)

    derivative_error = (error - previous_error) / dt
    
    control = Kp * error + Ki * integral_error + Kd * derivative_error

    flow_control = nominal_flow + control
    flow_control = max(min(flow_control, flow_max), flow_min)

    previous_error = error

    return flow_control, integral_error, previous_error, error



def PID_magnetic_coil_current(measured_oscillation, desired_oscillation, nominal_coil_current,
                              integral_error, previous_error, dt):
    # TODO: Implement Testbench and tune variables

    Kp = MAG_KP
    Ki = MAG_KI
    Kd = MAG_KD

    # Flow Safety
    coil_min = OUTER_MAG_MIN
    coil_max = OUTER_MAG_MAX

    # Integral Windup Prevention
    integral_min = None
    integral_max = None

    # Step 1: compute control error
    error = desired_oscillation - measured_oscillation

    # Step 2: update integral term
    integral_error += error * dt
    integral_error = max(min(integral_error, integral_max), integral_min)

    # Step 3: compute derivative term
    derivative_error = (error - previous_error) / dt

    # Step 4: PID formula
    correction = Kp * error + Ki * integral_error + Kd * derivative_error

    # Step 5: compute commanded magnetic coil current
    coil_command = nominal_coil_current + correction

    # Step 6: clamp coil command to safe range
    coil_command = max(min(coil_command, coil_max), coil_min)

    # Step 7: update memory for next call
    previous_error = error

    return coil_command, integral_error, previous_error, error

def clamp(value, min_value, max_value):
    return(max(min(value, max_value), min_value))

def desired_current(voltage, POWER_TARGET):
    safe_voltage = clamp(voltage, VOLTAGE_MIN, VOLTAGE_MAX)
    return POWER_TARGET / safe_voltage

def flow_rate(anode_flow, cathode_fraction = CATHODE_FRAC_NOMINAL):
    safe_fraction = clamp(cathode_fraction, CATHODE_FRAC_MIN, CATHODE_FRAC_MAX)

    return anode_flow * safe_fraction

def  main():

    return 0