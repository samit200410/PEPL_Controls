from http import client
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from dataclasses import asdict

import io
import json
import multiprocessing as mp
import numpy as np
import sys, struct, serial, time
import socket


# Global Variables

# ________________________________________________________
# Communication variables

# TCP Client Configuration
LABVIEW_IP = '10.0.0.1'
LABVIEW_PORT = 59704                                 # Set this to actual TCP port for receiving data from LabView
SOCKET_TIMEOUT = 7.0

# Keep False unless LabVIEW expects a flattened empty string
# argument for commands whose API argument is ""
SEND_EMPTY_ARG = False



HEADER_FMT = '>bi'                                  # byte, int
HEADER_LENGTH = struct.calcsize(HEADER_FMT)

# Command Constants
CMD_MAGNA_GET_READINGS = 0x12                       # Command 18
CMD_MAGNA_SET_CONTROL = 0x13                        # Command 19

CMD_ALICAT_GET_READINGS = 0x1B                      # Command 27
CMD_ALICAT_SET_CONTROL = 0x1C                       # Command 28

CMD_LAMBDA_GET_READINGS = 0x23                      # Command 35
CMD_LAMBDA_SET_CONTROL = 0x24                       # Command 36


# ________________________________________________________


# ________________________________________________________
# Shared variables for PID control

SETPOINT_FILE = "desired_current_setpoints.txt"
DEFAULT_DESIRED_CURRENT = 4.0
last_valid_desired_current = DEFAULT_DESIRED_CURRENT 
# tbd_current_vals = [10.0, 5.0, 13.0]

user_flag = False
measured_current = 0.0
nominal_flow = 5.0
integral_error_flow = 0.0
previous_error_flow = 0.0
dt = 0.5
# ________________________________________________________



# Data Structures 

@dataclass
class MagnaReadings:
    voltage: float
    current: float
    enabled: bool
    voltage_limit: float
    current_limit: float
    overvoltage_trip: float
    overcurrent_trip: float
    local_control: bool
    alarm: bool

@dataclass
class MagnaControl:
    voltage_limit: float
    current_limit: float
    overvoltage_trip: float
    overcurrent_trip: float
    enable: bool

@dataclass
class AlicatReadings:
    label: str
    gas: str
    setpoint: float
    setpoint_units: str
    mass_flow: float
    mass_flow_units: str
    pressure: float
    pressure_units: str
    temperature: float
    temperature_units: str
    volume_flow: float
    volume_flow_units: str
    valve_hold: bool

@dataclass
class AlicatControl:
    label: str
    setpoint: float
    units: str
    loop_control_variable: int=0            # U16 ENUM (unsigned word - 16 bits): 0 = Mass Flow, 1 = | Pressure |, 2 = Volume Flow
    valve_hold: bool = False

@dataclass
class LambdaReadings:
    label: str
    voltage: float
    current: float
    enable: bool
    voltage_limit: float
    current_limit: float
    overvoltage_protection: float
    remote_mode: int                        # U8 ENUM (unsigned byte - unsigned 8 bit int): 0 = Local, 1 = Remote, 2 = Local Lockout
    fault: bool

@dataclass
class LambdaControl:
    label: str
    voltage_limit: float
    current_limit: float
    overvoltage_protection: float
    enable: bool = False

@dataclass
class ManualControl:
    magna_supplies: MagnaControl
    alicat_supplies: list[AlicatControl]
    lambda_supplies: list[LambdaControl]
    send_magna: bool = True
    send_alicat: bool = True
    send_lambda: bool = True

@dataclass
class PIDControl:
    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_min: float
    integral_max: float

    integral: float = 0.0
    previous_error: float = 0.0
    first_call: bool = True

    def update(self, setpoint: float, measurement: float, nominal_output: float, dt: float) -> float:
        error = setpoint - measurement

        self.integral += error * dt
        self.integral = clamp(self.integral, self.integral_min, self.integral_max)

        if self.first_call:
            derivative = 0.0
            self.first_call = False
        else:
            derivative = (error - self.previous_error) / dt

        correction = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        output = nominal_output + correction
        output = clamp(output, self.output_min, self.output_max)

        self.previous_error = error
        return output



# Flattened Binary Reader/Writer Functions

class LabViewReader:
    
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def remaining(self) -> int:
        payload = self.payload
        offset = self.offset
        return len(payload) - offset
    
    def read_payload(self, n: int) -> bytes:
        payload = self.payload
        offset = self.offset

        if offset + n > len(payload):
            raise ValueError(
                f"Payload ended early. Need {n} bytes at offset {offset}, but only {self.remaining()} bytes remain."
            )
        
        out = payload[offset:offset+n]
        self.offset +=n
        return out

    # Signed 32 bit integer (big-endian)
    def i32(self) -> int:
        return struct.unpack(">i", self.read_payload(4))[0]
    
    # Unsigned 8 bit integer (big-endian)
    def u8(self) -> int:
        return struct.unpack(">B", self.read_payload(1))[0]
    
    # Unsigned 16 bit integer (big-endian)
    def u16(self) -> int:
        return struct.unpack(">H", self.read_payload(2))[0]
    
    # 64-bit float (big-endian)
    def f64(self) -> float:
        return struct.unpack(">d", self.read_payload(8))[0]
    
    # Boolean (big-endian)
    def boolean(self) -> bool:
        return self.u8() != 0
    
    # String (big-endian): 
    # First 4 bytes is length N, followed by N bytes of UTF-8 encoded string
    def string(self) -> str:
        length = self.i32()
        
        if length < 0:
            raise ValueError(
                f"Negative string length: {length}"
            )
        raw_bytes = self.read_payload(length)

        # Throw an error if the bytes aren't valid UTF-8
        # Replace invalid sequences with the Unicode replacement character instead of crashing
        return raw_bytes.decode("utf-8", errors="replace")
    
    def assert_consume_all(self, context: str) -> None:
        if self.remaining() != 0:
            extra = self.payload[self.offset:]
            raise ValueError(
                f"{context}: decoded payload but {self.remaining()} bytes remain unconsumed. "
                f"Number of Extra Bytes: {extra.hex(' ')}"
                )
    

class LabViewWriter:

    def __init__(self):
        self.value_types: list[bytes] = []

    def bytes(self) -> bytes:
        return b"".join(self.value_types)
    
    def i32(self, value: int) -> None:
        self.value_types.append(struct.pack(">i", int(value)))

    def u8(self, value: int) -> None:
        self.value_types.append(struct.pack(">B", int(value)))

    def u16(self, value: int) -> None:
        self.value_types.append(struct.pack(">H", int(value)))

    def f64(self, value: float) -> None:
        self.value_types.append(struct.pack(">d", float(value)))

    def boolean(self, value: bool) -> None:
        self.u8(1 if value else 0)

    def string(self, value: str) -> None:
        encoded = str(value).encode("utf-8")
        self.i32(len(encoded))
        self.value_types.append(encoded)


def flatten_empty_string() -> bytes:
    writer = LabViewWriter()
    writer.string("")
    return writer.bytes()


def empty_payload() -> bytes:
    return flatten_empty_string() if SEND_EMPTY_ARG else b""
    


# PEPL Lab Device Specific Unpacking Functions

def unpack_magna_readings(payload: bytes) -> MagnaReadings:
    reader = LabViewReader(payload)

    output = MagnaReadings(
        voltage = reader.f64(),
        current = reader.f64(),
        enabled = reader.boolean(),
        voltage_limit = reader.f64(),
        current_limit = reader.f64(),
        overvoltage_trip = reader.f64(),
        overcurrent_trip = reader.f64(),
        local_control = reader.boolean(),
        alarm = reader.boolean(),
    )

    reader.assert_consume_all("Magna Readings")
    return output


def unpack_alicat_readings(payload: bytes) -> list[AlicatReadings]:
    reader = LabViewReader(payload)
    
    # Array of Clusters
    n_controllers = reader.i32()

    output: list[AlicatReadings] = []
    for _ in range(n_controllers):
        output.append(
            AlicatReadings(
                label = reader.string(),
                gas = reader.string(),
                setpoint = reader.f64(),
                setpoint_units = reader.string(),
                mass_flow = reader.f64(),
                mass_flow_units = reader.string(),
                pressure = reader.f64(),
                pressure_units = reader.string(),
                temperature = reader.f64(),
                temperature_units = reader.string(),
                volume_flow = reader.f64(),
                volume_flow_units = reader.string(),
                valve_hold = reader.boolean(),
            )
        )

    reader.assert_consume_all("Alicat Readings")
    return output


def unpack_lambda_readings(payload: bytes) -> list[LambdaReadings]:
    reader = LabViewReader(payload)
    
    # Array of Clusters
    n_supplies = reader.i32()

    output: list[LambdaReadings] = []
    for _ in range(n_supplies):
        output.append(
            LambdaReadings(
                label = reader.string(),
                voltage = reader.f64(),
                current = reader.f64(),
                enable = reader.boolean(),
                voltage_limit = reader.f64(),
                current_limit = reader.f64(),
                overvoltage_protection = reader.f64(),
                remote_mode = reader.u8(),
                fault = reader.boolean(),
            )
        )

    reader.assert_consume_all("Lambda Readings")
    return output



# PEPL Lab Device Specific Packing Functions

def pack_magna_control(control: MagnaControl) -> bytes:
    writer = LabViewWriter()

    writer.f64(control.voltage_limit)
    writer.f64(control.current_limit)
    writer.f64(control.overvoltage_trip)
    writer.f64(control.overcurrent_trip)
    writer.boolean(control.enable)
    
    return writer.bytes()

def pack_alicat_control(controls: list[AlicatControl]) -> bytes:
    writer = LabViewWriter()

    writer.i32(len(controls))

    for c in controls:
        writer.string(c.label)
        writer.f64(c.setpoint)
        writer.string(c.units)
        writer.u16(c.loop_control_variable)
        writer.boolean(c.valve_hold)

    return writer.bytes()

def pack_lambda_control(controls: list[LambdaControl]) -> bytes:
    writer = LabViewWriter()

    writer.i32(len(controls))

    for c in controls:
        writer.string(c.label)
        writer.f64(c.voltage_limit)
        writer.f64(c.current_limit)
        writer.f64(c.overvoltage_protection)
        writer.boolean(c.enable)

    return writer.bytes()



# TCP Client 

def receive_from_labview(socket: socket.socket, n_bytes: int) -> bytes:
    data = b""

    while len(data) < n_bytes:
            packet = socket.recv(n_bytes - len(data))
            if not packet:
                raise ConnectionError(
                    f"Socket closed early. Expected {n_bytes} bytes, but only received {len(data)} bytes before connection closed."
                )
            
            data += packet

    return data


class LabViewClient:
    
    def __init__(self, host: str, port: int, timeout: float = SOCKET_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))
        print(
            f"Connected to LabVIEW at {self.host}:{self.port}"
        )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def send_packet(self, command_id: int, payload: bytes = b"") -> None:
        if self.socket is None:
            raise RuntimeError(
                "Socket is not connected."
            )
        
        header = struct.pack(HEADER_FMT, command_id, len(payload))
        self.socket.sendall(header + payload)

    def receive_packet(self, expected_command_id: int | None = None) -> tuple[int, bytes]:
        if self.socket is None:
            raise RuntimeError(
                "Socket is not connected."
            )
        
        header = receive_from_labview(self.socket, HEADER_LENGTH)
        command_id, payload_length = struct.unpack(HEADER_FMT, header)

        if payload_length < 0:
            raise ValueError(
                f"LabVIEW returned negative payload length: {payload_length}"
            )
        
        payload = receive_from_labview(self.socket, payload_length)

        if expected_command_id is not None and command_id != expected_command_id:
            raise ValueError(
                f"Unexpected command ID. Expected {expected_command_id}, but got {command_id}"
            )
        
        return command_id, payload
    
    def request(self, command_id: int, payload: bytes = b"") -> bytes:
        self.send_packet(command_id, payload)
        _, response_payload = self.receive_packet(expected_command_id = command_id)
        return response_payload
    


# API Handling

def check_empty_ack(name: str, response_payload: bytes) -> None:
    # LabVIEW acknowledges a set command with:
    # Payload length 0
    # Flattened Empty String: 00 00 00 00
    # Accept Both
    if response_payload in (b"", b"\x00\x00\x00\x00"):
        return
    print(
        f"Warning: {name} returned unexpected non-empty payload "
        f"({len(response_payload)}) bytes: {response_payload.hex(' ')}"
    )

def get_magna_readings(client: LabViewClient) -> MagnaReadings:
    payload = client.request(CMD_MAGNA_GET_READINGS, empty_payload())
    return unpack_magna_readings(payload)
    
def set_magna_control(client: LabViewClient, control: MagnaControl) -> None:
    response = client.request(CMD_MAGNA_SET_CONTROL, pack_magna_control(control))
    return check_empty_ack("Magna Set Control", response)

def get_alicat_readings(client: LabViewClient) -> list[AlicatReadings]:
    payload = client.request(CMD_ALICAT_GET_READINGS, empty_payload())
    return unpack_alicat_readings(payload)

def set_alicat_control(client: LabViewClient, control: list[AlicatControl]) -> None:
    response = client.request(CMD_ALICAT_SET_CONTROL, pack_alicat_control(control))
    return check_empty_ack("Alicat Set Control", response)

def get_lambda_readings(client: LabViewClient) -> list[LambdaReadings]:
    payload = client.request(CMD_LAMBDA_GET_READINGS, empty_payload())
    return unpack_lambda_readings(payload)

def set_lambda_control(client: LabViewClient, control: list[LambdaControl]) -> None:
    response = client.request(CMD_LAMBDA_SET_CONTROL, pack_lambda_control(control))
    return check_empty_ack("Lambda Set Controls", response)


# Debugging Interface



# PLACEHOLDER - Control Logic Goes Here

def control_placeholder(
        magna: MagnaReadings,
        alicat_controllers: list[AlicatReadings],
        lambda_supplies: list[LambdaReadings],
        manual_commands: ManualCommand
) -> ManualCommand:
    

    return manual_commands
        


# Main Loop

def main() -> None:

    client = LabViewClient(LABVIEW_IP, LABVIEW_PORT)

    try:
        client.connect()

        while True:

            # 1. Receive Fresh Readings - All LabVIEW Sections

            magna = get_magna_readings(client)
            alicat_controllers = get_alicat_readings(client)
            lambda_supplies = get_lambda_readings(client)

            # 2. Show Readings for Debugging
            #TODO


            # 3. Load Commands from Local Editable JSON File 
            # TODO: For Future Implementation --> Create GUI
            manual_commands = 

            # 4. Control Process to Modify Manual Commands
            # TODO: Implement Controls Algorithm
            proposed_commands = control_placeholder(
                magna = magna,
                alicat_controllers = alicat_controllers,
                lambda_supplies = lambda_supplies,
                manual_commands = manual_commands,
            )

            # 5. Send proposed commands back to LabVIEW
            if proposed_commands.send_magna:
                set_magna_control(client, proposed_commands.magna)

            if proposed_commands.send_alicat:
                set_alicat_control(client, proposed_commands.alicat_controllers)

            if proposed_commands.send_lambda:
                set_lambda_control(client, proposed_commands.lambda_supplies)

            time.sleep(0.5)     # Adjust Sleep Cycle Between Reads if Necessary


    except KeyboardInterrupt:
        print("\nStopped by user input.")

    finally:
        client.close()
        print("Socket closed.")


if __name__ == "__main__":
    main()










# Helper Functions

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)

