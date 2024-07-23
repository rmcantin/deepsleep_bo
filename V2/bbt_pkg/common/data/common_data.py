import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Properties:
    input: str = ''
    signals: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)


@dataclass
class SignalProperties:
    id: str
    num_channels: int
    num_elements: int
    sampling_rate: int
    num_fields: int = 0  # this is only set in bbt_loader (not in fit-importer)
    # optional
    subject_name: str = ''
    device_name: str = ''
    device_type: str = ''
    signal_type: str = ''
    signal_loc: List[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)  # REVIEW


@dataclass
class EventProperties:
    id: str
    num_channels: int
    num_elements: int
    init_values: List[int]
    # optional
    subject_name: str = ''
    device_name: str = ''
    device_type: str = ''
    config: dict = field(default_factory=dict)  # REVIEW


@dataclass
class Signal:
    ts: np.ndarray          # numpy array, 1 row (in us)
    fields: np.ndarray      # numpy array, 2 rows (sequence number, battery)
    values: np.ndarray      # numpy array (channels x samples)


@dataclass
class Event:
    ts: np.ndarray          # numpy array, 1 dimension (in seconds)
    values: np.ndarray      # numpy array (channels x samples)


@dataclass
class Timestamps:
    hat: np.ndarray  # numpy array, 1 dimension
    utc: np.ndarray  # numpy array, 1 dimension
