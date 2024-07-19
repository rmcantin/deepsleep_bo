import decimal
import numpy as np
from sys import platform
from typing import Union, List, Dict
from bbt_pkg.fit.core.utils import device_type
from bbt_pkg.common.data.common_data import Properties
from bbt_pkg.common.data.fit_data import Log


class EmptySignal(Exception):
    pass


def _signed_diff(seq1: int, seq2: int, dev_seq_bits: int) -> int:
    if seq2 <= seq1:
        seq2 = seq2 + 2 ** dev_seq_bits

    return seq2 - seq1


def seq_overflow(seq: np.ndarray, dev_seq_bits: int) -> np.ndarray:
    # corrects for a sequence number overflow
    # it checks for a difference <= 0 and assumes an overflow happened
    # returns the corrected sequence number
    idx_overflow = np.where(np.diff(seq) <= 0)[0]
    if len(idx_overflow) > 0:
        overflow_increase = [_signed_diff(seq[idx_overflow[i]],
                                          seq[idx_overflow[i] + 1],
                                          dev_seq_bits)
                             for i in range(len(idx_overflow))]

        for i in range(len(idx_overflow)):
            idx_1 = idx_overflow[i]
            idx_2 = idx_overflow[i] + 1
            offset = seq[idx_1] - seq[idx_2] + overflow_increase[i]
            seq[idx_2:] = seq[idx_2:] + offset

    return seq


def seq_overflow_ts(seq: np.ndarray, ts: np.ndarray, block_time_no_us: float,
                    dev_seq_bits: int, t0_us: Union[None, float]) -> np.ndarray:
    # corrects for a sequence number overflow by checking:
    # 1) the reception timestamps, and 2) the sequence number difference
    # inputs ts and block_time_no_us should be in us
    # returns the corrected sequence number
    t_overflow = 0.9 * block_time_no_us * (2 ** dev_seq_bits)  # adds some margin for the jitter

    if t0_us is not None:
        # check initial ts value against a t0 (needed in sd_bbt_combine)
        # it might recognize an initial overflow
        ts_measured = (ts[0] - t0_us) - (seq[0] * block_time_no_us)
        n_overflow = np.floor(ts_measured / t_overflow)
        if n_overflow > 0:
            seq = seq + n_overflow * (2 ** dev_seq_bits)
            #print('seq_overflow_ts warning, rare case, initial sequence overflow')

    # standard sequence overflow correction
    seq = seq_overflow(seq, dev_seq_bits)

    # check according to ts the number of overflows that happened (rare case)
    ts_measured = np.diff(ts) - (np.diff(seq) * block_time_no_us)
    n_overflow = np.floor(ts_measured / t_overflow)
    idx_overflow = np.where(n_overflow > 0)[0]
    if len(idx_overflow) > 0:
        #print('seq_overflow_ts warning, rare case, check sequence overflow')
        for i in range(len(idx_overflow)):
            idx_2 = idx_overflow[i] + 1
            seq[idx_2:] = seq[idx_2:] + (n_overflow[idx_overflow[i]] * (2 ** dev_seq_bits))

    return seq


def compute_block_time(seq: np.ndarray, ts: np.ndarray, block_time_no_us: float) -> float:
    # computes the block time by a linear regression of the sequence number and timestamp
    # this algorithm is enabled when there is enough input data
    # returns the estimated block time in us
    # note 1) for numerical reasons, ts for the regression algorithm is given in s
    # note 2) another method would be: (total_time / (number_blocks-1))
    num_points = len(seq)
    total_time_s = 1e-6 * (ts[-1] - ts[0])
    if (num_points >= 320) and (total_time_s > 30):
        block_time_us = 1e6 * _bbt_regression(seq, 1e-6 * ts)
    else:
        block_time_us = block_time_no_us

    return block_time_us


def _bbt_regression(seq: np.ndarray, ts: np.ndarray) -> float:
    # returns the slope of a linear regression
    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    num_points = len(seq)
    for i in range(num_points):
        x = seq[i]
        y = ts[i]
        sx = sx + x
        sy = sy + y
        sxx = sxx + (x * x)
        sxy = sxy + (x * y)
    return (num_points * sxy - sx * sy) / (num_points * sxx - sx * sx)


def flight_time(device_type: str) -> float:
    # returns the flight type of the bbt device (in us)
    flight_time_ms = 35 * 1e3
    if device_type == 'bth_cap_32':
        flight_time_ms = 40 * 1e3
    elif device_type == 'ble_cap_05':
        flight_time_ms = 25 * 1e3
    return flight_time_ms


def round_int(value: float) -> int:
    # rounds a float to integer as c++ and matlab do
    return int(decimal.Decimal(value).to_integral_value(rounding=decimal.ROUND_HALF_UP))


def round_float(value: float) -> np.double:
    # rounds a float to integer as c++ and matlab do
    return np.float64(decimal.Decimal(value).to_integral_value(rounding=decimal.ROUND_HALF_UP))


def enabled_signals(log: Log) -> List[str]:
    return [signal_id for signal_id in log.signals
            if log.signals[signal_id].enabled]


def disabled_signals(log: Log) -> List[str]:
    return [signal_id for signal_id in log.signals
            if not log.signals[signal_id].enabled]


def enabled_signals_by_class(log: Log, properties: Properties,
                             device_class: device_type.DeviceClass) -> List[str]:
    return [signal_id for signal_id in log.signals
            if log.signals[signal_id].enabled
            and device_type.is_class(properties.signals[signal_id].device_type, device_class)]


def signals_by_class(properties: Properties, device_class: device_type.DeviceClass) -> List[str]:
    return [signal_id for signal_id in properties.signals
            if device_type.is_class(properties.signals[signal_id].device_type, device_class)]


def signals_per_device(properties: Properties, signal_id_list: List[str]) -> Dict[str, List[str]]:
    dev_signals = {}
    for signal_id in signal_id_list:
        device_name = properties.signals[signal_id].device_name
        if device_name not in dev_signals:
            dev_signals[device_name] = [signal_id]
        else:
            dev_signals[device_name].append(signal_id)
    return dev_signals


def norm_path(file_path: str) -> str:
    output_file_path = file_path
    if platform == "Linux":
        output_file_path = file_path.replace('\\', '//')
    return output_file_path
