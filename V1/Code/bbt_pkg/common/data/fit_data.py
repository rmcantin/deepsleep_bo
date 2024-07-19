import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any
from bbt_pkg.fit.core.utils import device_type


class AdjustMethods(Enum):
    LITE = 'lite'
    FULL = 'full'

    @staticmethod
    def from_str(label: str):
        if label == 'lite':
            return AdjustMethods.LITE
        elif label == 'full':
            return AdjustMethods.FULL
        else:
            raise NotImplementedError


class DriftCorrMethods(Enum):
    NO = 'no'
    SOFT = 'soft'
    COMPLETE = 'complete'

    @staticmethod
    def from_str(label: str):
        if label == 'no':
            return DriftCorrMethods.NO
        elif label == 'soft':
            return DriftCorrMethods.SOFT
        elif label == 'complete':
            return DriftCorrMethods.COMPLETE
        else:
            raise NotImplementedError


@dataclass(frozen=True)
class FitCfgData:
    sd_enabled: bool
    debug: bool
    dev_seq_bits: int
    drift_corr_method: str
    adjust_method: str
    dev_type: Dict[str, device_type.DeviceClass] = field(default_factory=dict)
    check_sd: bool = False


@dataclass
class LogUTC:
    enabled: bool = False
    stable: bool = False
    t0: float = float('nan')
    tn: float = float('nan')


@dataclass
class SignalLog:
    enabled: bool = False
    error_flag: int = 0
    error_message: str = ''
    sd_data_read: bool = False
    sd_sync_read: bool = False
    t0: float = float('nan')
    tn: float = float('nan')
    pad_ini: int = 0
    pad_end: int = 0
    sr_no: int = 0
    sr_st: float = 0
    utc: LogUTC = field(default_factory=LogUTC)
    private: Dict[str, Any] = field(default_factory=dict)  # PRIVATE


@dataclass
class LogSystem:
    version: float = -1
    method: str = ''
    adjust: str = ''


@dataclass
class Log:
    ok: bool = False
    t0: float = float('nan')
    tn: float = float('nan')
    utc: LogUTC = field(default_factory=LogUTC)
    signals: Dict[str, SignalLog] = field(default_factory=dict)
    system: LogSystem = field(default_factory=LogSystem)

    def items(self):
        yield from self.__dict__.items()


@dataclass
class FitSignal:
    ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype='float64'))  # numpy array, 1 dimension (in us)
    ts_corr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype='float64'))  # numpy array, 1 dimension (in us)
    seq: np.ndarray = field(default_factory=lambda: np.zeros(0))  # numpy array, 1 dimension (sequence number)
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))  # numpy array (channels x samples)


@dataclass
class FitEvent:
    onset: np.ndarray = field(default_factory=lambda: np.zeros(0))  # numpy array, 1 dimension (sequence number)
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))  # numpy array (channels x samples)
