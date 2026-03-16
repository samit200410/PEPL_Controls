import numpy as np
import threading, sys, time, struct, serial

# TODO: Check how packets are sent from LabView
# CRC8 function for checking packet integrity from LabView
def compute_crc8(data: bytes) -> int:
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc

def send_packet(ser, packet):
    # TODO: Implement packet sending to LabView
    pass


def receive_packet(ser):
    # TODO: Implement packet receiving from LabView
    pass

def PID_threadspawner():
    # TODO: Implement thread spawner for PID control of two processes
    
    # Start thread for discharge current controls

    # Start thread for magnetic coil current control
    pass

def PID_discharge_current():
    #TODO: Implement PID control for discharge current
    pass

def PID_magnetic_coil_current():
    #TODO: Implement PID control for magnetic coil current
    pass

def  main():

    return 0