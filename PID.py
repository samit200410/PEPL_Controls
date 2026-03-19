import numpy as np
import sys, time, struct, serial
import multiprocessing as mp

SERIAL_PORT = ""
SYNC = 0xAA
PAYLOAD_SIZE = -1  # TODO: Set this to the actual payload size from LabView
READ_FMT = None # TODO: Set this to the actual struct format string for unpacking LabView packets
WRITE_FMT = None # TODO: Set this to the actual struct format string for packing packets to LabView

# Shared Data
latest_packet = None

# Locks for shared data
reader_lock = mp.Lock()
writer_lock = mp.Lock()


# TODO: Check how packets are sent from LabView
# CRC8 function for checking packet integrity from LabView
def compute_crc8(data: bytes) -> int:
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc

def reader_thread():
    # TODO: Implement reader thread for handling packet send and receive
    global latest_packet
    try:
        while(True):
            b = serial.read(1)
            if not b:
                continue
            if b[0] != SYNC:
                continue
            payload = serial.read(PAYLOAD_SIZE)
            
            crc_bit = serial.read(1)
            if len(crc_bit) != 1:
                continue
            if compute_crc8(payload) != crc_bit[0]:
                continue

            try:
                pkt = struct.unpack(READ_FMT, payload)
            except struct.error:
                print("What the freak?")
                continue

            with lock:
                latest_packet = pkt

    except Exception:
        print("Error in reader thread")

def writer_thread():
    # TODO: Implement writer thread for sending packets to LabView
    
    pass

def PID_threadspawner():
    # TODO: Implement thread spawner for PID control of two processes

    # Start thread for packet send and receive

    
    # Start thread for discharge current controls

    # Start thread for magnetic coil current control
    pass

def PID_discharge_current(measured_current, desired_current, nominal_flow,
                          integral_error, previous_error, dt):
    
    # TODO: Implement Testbench and tune variables

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


def PID_magnetic_coil_current(measured_oscillation, desired_oscillation, nominal_coil_current,
                              integral_error, previous_error, dt):
    # TODO: Implement Testbench and tune variables

    Kp = None
    Ki = None
    Kd = None

    # Flow Safety
    coil_min = None
    coil_max = None

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

def  main():

    return 0