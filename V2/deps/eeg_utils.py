import numpy as np
from bbt import Signal


class CircularBuffer:
    def __init__(self, num_channels: int, max_length: int):
        self.num_samples = 0
        self.max_length = max_length
        self.buffer = np.zeros((num_channels, self.max_length))

    def add(self, data: np.ndarray):
        num_samples = data.shape[1]

        self.buffer[:, :-num_samples] = self.buffer[:, num_samples:]
        self.buffer[:, -num_samples:] = data
        self.num_samples += num_samples
        if self.num_samples > self.max_length:
            self.num_samples = self.max_length

    def fill(self, value: float):
        self.buffer.fill(value)

    def get_last(self, num_samples: int) -> np.ndarray:
        return self.buffer[:, -num_samples:]


def eeg_extract_signal(properties, signals) -> (np.ndarray, int, float):
    signal_type = 'eeg'
    properties_eeg = dict(filter(lambda item: item[1].signal_type == signal_type, properties.signals.items()))
    if properties_eeg.keys() is None:
        print('error: signal type not found {}'.format(signal_type))
    else:
        signal_id = next(iter(properties_eeg.keys()))
        device_name = properties.signals[signal_id].device_name

    eeg_sampling_rate = properties.signals[signal_id].sampling_rate
    eeg_t0 = signals[signal_id].ts[0] - 31.25*1e3

    return signals[signal_id].values, eeg_sampling_rate, eeg_t0


def eeg_extract(properties, signals) -> (Signal, int, float):
    signal_type = 'eeg'
    properties_eeg = dict(filter(lambda item: item[1].signal_type == signal_type, properties.signals.items()))
    if properties_eeg.keys() is None:
        print('error: signal type not found {}'.format(signal_type))
    else:
        signal_id = next(iter(properties_eeg.keys()))
        device_name = properties.signals[signal_id].device_name

    eeg_sampling_rate = properties.signals[signal_id].sampling_rate
    eeg_t0 = signals[signal_id].ts[0] - 31.25*1e3

    return signals[signal_id], eeg_sampling_rate, eeg_t0

