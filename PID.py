from http import client
import io

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

ANODE_FLOW_MIN = 0.0 # 60.0
ANODE_FLOW_MAX = 120.0

CATHODE_FRAC_MIN = 0.05
CATHODE_FRAC_NOMINAL = 0.07
CATHODE_FRAC_MAX = 0.10
# ----------------------------


# ----------------------------
# PID Turning

# Mass Flow
FLOW_KP = 0.5
FLOW_KI = 0.0
FLOW_KD = 0.0

# Magnetic Field
MAG_KP = None
MAG_KI = None
MAG_KD = None
# ----------------------------



# TCP Server Configuration (TX)
HOST = socket.gethostbyname(socket.gethostname()) #
print("Host IP: ", HOST)
TCP_PORT_RX = 54709         # Set this to actual TCP port for transmitting data to LabView
TCP_RX = (HOST, TCP_PORT_RX)

# TCP Client Configuration (RX)
LABVIEW_IP = '10.0.0.1'
TCP_PORT_TX = 59704         # Set this to actual TCP port for receiving data from LabView
TCP_TX = (LABVIEW_IP, TCP_PORT_TX)

# Message Protocol Constants
HEADER = 4
COMMAND = 1  
LENGTH = 4

# Struct format for packing/unpacking data --> Big-Endian Order
struct_fmt = '>bi2d?4d2?'           # Example: byte, int, 2 doubles, bool, 4 doubles, 2 bools
exp_length = struct.calcsize(struct_fmt)
print("Expected Packet Length: ", exp_length)

# Struct client commands --> Big-Endian Order
cmd_fmt = '>bi4d?'                  # Example: byte, int, 4 doubles, bool
cmd_length = struct.calcsize(cmd_fmt)
print("Command Packet Length: ", cmd_length)

# Global shared variables for PID control
user_flag = False
measured_current = 0.0
desired_current_value = 4.0
tbd_current_vals = [10.0, 5.0, 13.0]
nominal_flow = 5.0
integral_error_flow = 0.0
previous_error_flow = 0.0
dt = 0.5

# Commands
CMD_GET_DATA = 0x11
CMD_SET_DATA = 0x10

# Shared Data
latest_packet = None

# Locks for shared data
reader_lock = mp.Lock()
writer_lock = mp.Lock()


def TCP_server_thread(conn, addr):
    with conn:
        print('Connected by', addr)
        while True:
            dat = conn.recv(exp_length)
            if not dat: break

            header, en, v_lim, i_lim = struct.unpack(struct_fmt, dat)
            print(f"Received header: {header}, enabled: {en}, voltage limit: {v_lim}, current limit: {i_lim}")

            # cmd = conn.recv(COMMAND)
            # if not cmd: break
            # msg_cmd = cmd.decode('utf-8').strip()
            # dat_len = conn.recv(LENGTH)
            # if not dat_len: break
            # msg_len = dat_len.decode('utf-8').strip()
            # data = conn.recv(int(msg_len))
            # if not data: break
            # print(f"Received command: {msg_cmd}, data length: {msg_len}, data: {data}")


# def Receive_Terminal_Input():
#     while True:
#         user_input = input("Enter current value (or 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             print("Exiting terminal input thread.")
#             break
#         else:
#             global new_desired_current_value
#             new_desired_current_value = float(user_input)
#             user_flag = True
            
        

def PID_threadspawner():

    # user_thread = mp.Process(target=Receive_Terminal_Input)
    # user_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        
        print("Communicating with LabVIEW...")
        client_socket.settimeout(7.0)
        
        
        try:
            client_socket.connect(TCP_TX)
        except TimeoutError:
            print("Connection to LabVIEW timed out. Please check the connection and try again.")
            return

        # Send data to Labview
        try:
            count = 0
            while True:
                idx = int(count / 15)
                # global user_flag
                # if user_flag:
                #     global desired_current_value
                #     desired_current_value = new_desired_current_value
                #     user_flag = False
                # 1) Get = command 17 --> Structure: byte, int, 2 doubles, bool, 4 doubles, 2 bools
                print("Requesting data from LabVIEW...")
                packet = struct.pack('>bi', CMD_GET_DATA, 0x00000000)
                client_socket.sendall(packet)
                ack = client_socket.recv(1024)
                cmd_ret, length, voltage, current, enabled, voltage_limit, current_limit, voltage_trip, current_trip, local_ctrl, alarm = struct.unpack(struct_fmt, ack)
                if cmd_ret == CMD_GET_DATA:
                    print("cmd:", cmd_ret, "\nlength:", length, "\nvoltage: ", voltage, "\ncurrent: ", current, "\nenabled: ", enabled,
                        "\nvoltage_limit: ", voltage_limit, "\ncurrent_limit: ", current_limit, "\nvoltage_trip: ", voltage_trip, "\ncurrent_trip: ", current_trip, "\nLocal Control: ", local_ctrl, "\nAlarm: ", alarm)
                
                shutdown = False
                if voltage > 25:
                    print("Voltage exceeds safe threshold! Initiating shutdown...")
                    shutdown = True

                # measured_current = current
                global integral_error_flow, previous_error_flow
                flow_control, integral_error_flow, previous_error_flow = PID_discharge_current(measured_current, tbd_current_vals[idx], nominal_flow, integral_error_flow, previous_error_flow, dt)
                print("Computed flow control: ", flow_control, "\nIntegral Error: ", integral_error_flow, "\nPrevious Error: ", previous_error_flow)
                


                # 2) Set = command 16 --> Structure: byte, int, 4 doubles, bool
                # CMD, length, voltage_lim, current_lim, voltage_trp, current_trp, enable
                print("\nSending data to LabVIEW...")
                
                packet = struct.pack(cmd_fmt, CMD_SET_DATA, 0x00000021, flow_control, 15.0, 50.0, 50.0, not shutdown)
                print("Packed data to send: ", packet)
                client_socket.sendall(packet)
                ack = client_socket.recv(1024)
                cmd, length = struct.unpack('>bi', ack)
                print("cmd: ", cmd, "\nlength: ", length)

                count += 1
                
                time.sleep(0.5)
            
    
        except Exception as e:
            print("Generic Error:", e)
        finally:
            client_socket.close()
    
    # user_thread.join()

    return

    # TODO: Implement thread spawner for PID control of two processes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(TCP_RX)
        s.listen(1)
        conn, addr = s.accept()

    # Start thread for packet send and receive
    # reader_thread = mp.Process(target=TCP_server_thread, args=(conn, addr))
    # reader_thread.start()
    
    # Start thread for discharge current controls
    # discharge_current_thread = mp.Process(target=PID_discharge_current, args=(...)) # TODO: Fill in arguments
    # discharge_current_thread.start()

    # Start thread for magnetic coil current control
    # magnetic_coil_thread = mp.Process(target=PID_magnetic_coil_current, args=(...)) # TODO: Fill in arguments
    # magnetic_coil_thread.start()

    # reader_thread.join()
    # discharge_current_thread.join()
    # magnetic_coil_thread.join()



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

    try: 
        error = desired_current - measured_current

        integral_error += error * dt
        integral_error = max(min(integral_error, integral_max), integral_min)

        derivative_error = (error - previous_error) / dt
        
        control = Kp * error + Ki * integral_error + Kd * derivative_error
        print("control: ", control)

        flow_control = nominal_flow + control
        flow_control = max(min(flow_control, flow_max), flow_min)

        previous_error = error
    except Exception as e:
        print("Error in PID_discharge_current: ", e)
        flow_control = nominal_flow

    return flow_control, integral_error, previous_error



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

if __name__ == "__main__":

    PID_threadspawner()

    print("PID control threads have completed execution.")

    