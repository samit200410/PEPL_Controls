from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import copy, socket, struct, time


# GLOBAL VARIABLES

# ---------------------------------------------------------------------------
# Communication variables

# TCP Client Configuration
LABVIEW_IP = '10.0.0.1'
LABVIEW_PORT = 59704                                 # Set this to actual TCP port for receiving data from LabView
SOCKET_TIMEOUT = 7.0

# Keep False unless LabVIEW expects a flattened empty string argument for ""
SEND_EMPTY_ARG = False

HEADER_FMT = '>bi'                                  # byte command_id, signed int payload_length
HEADER_LENGTH = struct.calcsize(HEADER_FMT)

# Command Constants from API_Short.xlsx
CMD_MAGNA_GET_READINGS = 0x12                       # Command 18
CMD_MAGNA_SET_CONTROL = 0x13                        # Command 19

CMD_ALICAT_GET_READINGS = 0x1B                      # Command 27
CMD_ALICAT_SET_CONTROL = 0x1C                       # Command 28

CMD_LAMBDA_GET_READINGS = 0x23                      # Command 35
CMD_LAMBDA_SET_CONTROL = 0x24                       # Command 36

CMD_OSCOPE_GET_READINGS = 0x2B                      # Command 43
                                                    
CMD_DMM_GET_READINGS = 0x31                         # Command 49

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# Safe Operating Ranges for H9 HET
# DISCHARGE_VOLTAGE_MIN = 300.0                       # Volts
# DISCHARGE_VOLTAGE_MAX = 400.0                       # Volts

# DISCHARGE_CURRENT_MIN = 15.0                        # Amps
# DISCHARGE_CURRENT_MAX = 30.0                        # Amps

# MASS_FLOW_MIN = 10.0                                # mg/sec
# MASS_FLOW_MIN = 20.0                                # mg/sec

# MAGNET_PERCENT_MIN = 75.0                           # %
# MAGNET_PERCENT_MAX = 125.0                          # %
# ---------------------------------------------------------------------------



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
class OscopeAxis:
    increment: float
    origin: float
    reference: int

@dataclass
class OscopeWaveform:
    x: OscopeAxis
    y: OscopeAxis
    data: list[int]

    # Return physical x-axis values for each waveform point.
    # For Keysight-style waveform scaling: 
    #           time[i] = (i - x_reference) * x_increment + x_origin

    def time_values(self) -> list[float]:
        
        return [
            (i - self.x.reference) * self.x.increment + self.x.origin
            for i in range(len(self.data))
        ]

    # For Keysight-style waveform scaling:
    #           ignal[i] = (raw[i] - y_reference) * y_increment + y_origin

    def y_values(self) -> list[float]:

        return [
            (float(raw_point) - self.y.reference) * self.y.increment + self.y.origin
            for raw_point in self.data
        ]
    
    @property
    def sample_rate_hz(self) -> float | None:
        if self.x.increment == 0:
            return None
        return 1.0 / abs(self.x.increment)
    
    @property
    def duration(self) -> float:
        if len(self.data) <= 1:
            return 0.0
        return(len(self.data) - 1) * abs(self.x.increment)
        
@dataclass
class OscopeReadings:
    label: str
    peak_to_peak: float
    rms: float
    average: float
    wavform: OscopeWaveform

@dataclass
class DeviceCommands:
    magna_supplies: MagnaControl
    alicat_supplies: list[AlicatControl]
    lambda_supplies: list[LambdaControl]



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
    
    # Unsigned 32 bit integer (big_endian)
    def u32(self) -> int:
        return struct.upnack(">I", self.read_payload(4))[0]
    
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
    

    def array_length(self, context:str) -> int:
        length = self.i32()
        if length < 0:
            raise ValueError(f"{context}: negative array length = {length}")
        return length

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

    def u32(self, value: int) -> None:
        self.value_types.append(struct.pack(">I", int(value)))

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
    n_controllers = reader.array_length("Alicat Readings")

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
    n_supplies = reader.array_length("Lambda Readings")

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


def unpack_oscope_waveform(reader: LabViewReader) -> OscopeWaveform:
    # Waveform Cluster from LabVIEW:
    #   X Cluster: X increment (Double), X origin (Double), X Reference (U32)
    #   Y Cluster: Y increment (Double), Y origin (Double), Y Reference (U32)
    #   Data: 1D array (U16 points)
    
    x_axis = OscopeAxis(
        increment = reader.f64(),
        origin = reader.f64(),
        reference = reader.u32(),
    )

    y_axis = OscopeAxis(
        increment = reader.f64(),
        origin = reader.f64(),
        reference = reader.u32(),
    )

    n_points = reader.array_length("Oscope Wavefrom Data")
    data = [reader.u16() for _ in range(n_points)]

    return OscopeWaveform(x = x_axis, y = y_axis, data = data)

def unpack_oscope_readings(payload: bytes) -> list[OscopeReadings]:
    reader = LabViewReader(payload)

    # 1-D Array of Keysight O-Scope Single Readings.ctl" Clusters
    n_readings = reader.array_length("Oscope Readings")

    output: list[OscopeReadings] = []

    for _ in range(n_readings):
        output.append(
            OscopeReadings(
                label = reader.string(),
                peak_to_peak = reader.f64(),
                rms = reader.f64(),
                average = reader.f64(),
                wavform = unpack_oscope_waveform(reader),
            )
        )

    reader.assert_consume_all("Oscope Readings")

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
                    f"Socket closed early. Expected {n_bytes} bytes. "
                    f"Only received {len(data)} bytes before connection closed."
                )
            
            data += packet

    return data

class LabViewClient:
    
    def __init__(self,
                 host: str = LABVIEW_IP,
                 port: int = LABVIEW_PORT,
                 timeout: float = SOCKET_TIMEOUT
                ):
        
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

    def __enter__(self) -> "LabViewClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

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

def get_oscope_readings(client: LabViewClient) -> list[OscopeReadings]:
    payload = client.request(CMD_OSCOPE_GET_READINGS, empty_payload())
    return unpack_oscope_readings(payload)





















def get_all_readings(client: LabViewClient, *, include_oscope: bool = True) -> dict[str, Any]:




# "GET" Measurments & Make a Shipping Packet Acceptable by Control Model

def get_measurements_for_model(client: LabViewClient) -> dict[str, Any]:

    # "GET" all LabVIEW measurements and return a plain Python dictionary    
    magna_supplies = get_magna_readings(client)
    alicat_supplies = get_alicat_readings(client)
    lambda_supplies = get_lambda_readings(client)
    oscope_readings = get_oscope_readings(client)

    return {
        "timestamp_s": time.time(),
        "magna": asdict(magna_supplies),
        "alicat": [asdict(item) for item in alicat_supplies],
        "lambda": [asdict(item) for item in lambda_supplies],
        "oscope": [asdict(item) for item in oscope_readings],

    }


# Diffusiuon Model Interface

# Wants flat dictionary with physical control names:
#
#   {
#       "anode_flow_rate": float,
#       "magnet_current_outer": float,
#       "magnet_current_inner": float,
#       "discharge_voltage": float,
#       "cathode_flow_rate": float,
#   }

# The aliases to connect those physical names to the exact LabVIEW labels

ANODE_ALICAT_LABEL_ALIAS = (

)

CATHODE_ALICAT_LABEL_ALIASES = (

)

OUTER_MAGNET_LAMBDA_LABEL_ALIASES = (
 
)

INNER_MAGNET_LAMBDA_LABEL_ALIASES = (

)











def pack_control_inputs_for_model(client: LabViewClient) -> dict[str, float]:

    measurements = get_measurements_for_model(client)
    
    magna_raw = measurements["magna"]
    alicat_raw = measurements["alicat"]
    lambda_raw = measurements["lambda"]

    # Dynamically search lists of dicts by hardware label
    def find_by_label(items: list[dict[str, Any]], label_query: str) -> dict[str, Any]:
        for item in items:
            if label_query.lower() in str(item.get("label", "")).lower():
                return item
        raise ValueError(f"Could not find hardware with label containing '{label_query}'")

    try:
        # 2. Attempt to resolve devices dynamically by their configured LabVIEW labels
        anode_mfc = find_by_label(alicat_raw, "anode")
        cathode_mfc = find_by_label(alicat_raw, "cathode")
        outer_magnet = find_by_label(lambda_raw, "outer")
        inner_magnet = find_by_label(lambda_raw, "inner")
        
        return {
            "anode_flow_rate": float(anode_mfc["mass_flow"]),
            "magnet_current_outer": float(outer_magnet["current"]),
            "magnet_current_inner": float(inner_magnet["current"]),
            "discharge_voltage": float(magna_raw["voltage"]),
            "cathode_flow_rate": float(cathode_mfc["mass_flow"]),
        }

    except ValueError as e:
        # 3. Fallback Block: If labels don't match, fall back to sequential list indexing
        print(f"Warning: Label lookup failed ({e}). Falling back to array order indexing.")
        
        return {
            "anode_flow_rate": float(alicat_raw[0]["mass_flow"]) if len(alicat_raw) > 0 else 0.0,
            "magnet_current_outer": float(lambda_raw[0]["current"]) if len(lambda_raw) > 0 else 0.0,
            "magnet_current_inner": float(lambda_raw[1]["current"]) if len(lambda_raw) > 1 else 0.0,
            "discharge_voltage": float(magna_raw["voltage"]),
            "cathode_flow_rate": float(alicat_raw[1]["mass_flow"]) if len(alicat_raw) > 1 else 0.0,
        }










# Fetch Control Model Setpoints & "SET" them in LabVIEW

def commands_from_measurements(measurements: dict[str, Any]) -> DeviceCommands:
    magna_raw = measurements["magna"]
    alicat_raw = measurements["alicat"]
    lambda_raw = measurements["lambda"]

    magna_supplies = MagnaControl(
        voltage_limit = float(magna_raw["voltage_limit"]),
        current_limit = float(magna_raw["current_limit"]),
        overvoltage_trip = float(magna_raw["overvoltage_trip"]),
        overcurrent_trip = float(magna_raw["overcurrent_trip"]),
        enable = bool(magna_raw["enable"]),
    )

    alicat_supplies = [
        AlicatControl(
            label = str(item["label"]),
            setpoint = float(item["setpoint"]),
            units = str(item["setpoint_units"]),
            loop_control_variable = 0,
            valve_hold = bool(item["valve_hold"]),
        )

        for item in alicat_raw

    ]

    lambda_supplies = [
        LambdaControl(
            label = str(item["label"]),
            voltage_limit = float(item["voltage_limit"]),
            current_limit = float(item["current_limit"]),
            overvoltage_protection = float(item["overvoltage_protection"]),
            enable = bool(item["enable"]),
        )

        for item in lambda_raw

    ]

    return DeviceCommands(magna_supplies = magna_supplies , alicat_supplies = alicat_supplies, lambda_supplies = lambda_supplies)


def _find_alicat(commands: DeviceCommands, label: str) -> AlicatControl:
    for item in commands.alicat_supplies:
        if item.label == label:
            return item
    raise ValueError(f"Could not find Alicat controller with label {label!r}.")


def _find_lambda(commands: DeviceCommands, label: str) -> LambdaControl:
    for item in commands.lambda_supplies:
        if item.label == label:
            return item
    raise ValueError(f"Could not find Lambda supply with label {label!r}.")


def apply_model_setpoints(commands: DeviceCommands, model_setpoints: dict[str, Any]) -> DeviceCommands:

    updated = copy.deepcopy(commands)

    # Magna-Power discharge supply updates.
    magna_sp = model_setpoints.get("magna", {})
    if "voltage_limit" in magna_sp:
        updated.magna_supplies.voltage_limit = float(magna_sp["voltage_limit"])
    if "current_limit" in magna_sp:
        updated.magna_supplies.current_limit = float(magna_sp["current_limit"])
    if "overvoltage_trip" in magna_sp:
        updated.magna_supplies.overvoltage_trip = float(magna_sp["overvoltage_trip"])
    if "overcurrent_trip" in magna_sp:
        updated.magna_supplies.overcurrent_trip = float(magna_sp["overcurrent_trip"])
    if "enable" in magna_sp:
        updated.magna_supplies.enable = bool(magna_sp["enable"])

    # Alicat MFC updates. Send the full array, but only modify listed labels.
    for alicat_sp in model_setpoints.get("alicat", []):
        label = str(alicat_sp["label"])
        target = _find_alicat(updated, label)
        if "setpoint" in alicat_sp:
            target.setpoint = float(alicat_sp["setpoint"])
        if "units" in alicat_sp:
            target.units = str(alicat_sp["units"])
        if "loop_control_variable" in alicat_sp:
            target.loop_control_variable = int(alicat_sp["loop_control_variable"])
        if "valve_hold" in alicat_sp:
            target.valve_hold = bool(alicat_sp["valve_hold"])

    # Lambda auxiliary supply updates. Send the full array, but only modify listed labels.
    for lambda_sp in model_setpoints.get("lambda", []):
        label = str(lambda_sp["label"])
        target = _find_lambda(updated, label)
        if "voltage_limit" in lambda_sp:
            target.voltage_limit = float(lambda_sp["voltage_limit"])
        if "current_limit" in lambda_sp:
            target.current_limit = float(lambda_sp["current_limit"])
        if "overvoltage_protection" in lambda_sp:
            target.overvoltage_protection = float(lambda_sp["overvoltage_protection"])
        if "enable" in lambda_sp:
            target.enable = bool(lambda_sp["enable"])

    return updated


def send_model_setpoints_to_labview(client: LabViewClient, model_setpoints: dict[str, Any], latest_measurements: dict[str, Any] | None = None) -> DeviceCommands:

    if latest_measurements is None:
        latest_measurements = get_measurements_for_model(client)

    base_commands = commands_from_measurements(latest_measurements)
    final_commands = apply_model_setpoints(base_commands, model_setpoints)

    set_magna_control(client, final_commands.magna_supplies)
    set_alicat_control(client, final_commands.alicat_supplies)
    set_lambda_control(client, final_commands.lambda_supplies)

    return final_commands
    












