import numpy as np
from scipy import signal
from deps.eeg_utils import CircularBuffer
from bbt_pkg.fit.core.utils import fit_utils
from deps.power import welch_bbt
from bbt_pkg.fit.core.utils.fit_utils import _signed_diff


def compute_welch_psd(eeg: np.array, window_type: str, window_size: int, noverlap: int, nfft: int,
                      sampling_rate: int, channels: list, max_freq: int, log10: bool = True):

    # Example of previous uses in Bitbrain
    # window_type = 'hamming'
    # window_size = sampling_rate
    # noverlap = int(sampling_rate/2)
    # nfft = 4 * sampling_rate
    # fs = sampling_rate
    # channels = list of integers with the channels positions to be computed
    # max_freq = integers with the maximum freq to be calculated (adapt to your requirements)

    psd, freq = welch_bbt(x=eeg, window_type=window_type, window_size=window_size,
                          noverlap=noverlap, nfft=nfft, fs=sampling_rate, channels=channels,
                          max_freq=max_freq, average_freq_bins=True, log10=log10)
    return psd, freq


class FrequenciesDetector:

    def __init__(self, selected_channel: int, sampling_rate: int, delta_band: list = [0.5, 4], delta_th: float = 20,
                 beta_band: list = [17, 30], beta_th: float = 20, buffer_size: int = 4, delta_decoder_on: bool = True,
                 beta_decoder_on: bool = True):

        # channel
        self.num_channels = 1
        self.ch = selected_channel - 1

        # buffering
        self.srate = sampling_rate
        self.update_time = 1
        self.buffer_size_secs = buffer_size
        self.buffer_size_samples = buffer_size * self.srate
        self.buffer_eeg = CircularBuffer(1, self.buffer_size_samples)
        self.buffer_seq = CircularBuffer(1, self.buffer_size_samples)

        # variables
        self.block_time = 1e-3 * 31.25
        update_time = 1
        self.update_blocks = update_time / self.block_time
        self.blocks_counter = 0

        # variables, inhibition time
        self.waiting_time = 10
        self._waiting = False
        self._waiting_sequence_num = -1

        # variables, buffering
        self.previous_seq = -1

        # variables delta
        self.delta_band = delta_band
        self.delta_th = delta_th
        self.delta = [-100000000000000000000000.0]
        self.delta_on = False
        self.delta_decoder_on = delta_decoder_on
        self.deltad_status = np.zeros((6, 1))

        # variables beta
        self.beta_band = beta_band
        self.beta_th = beta_th
        self.beta = [-100000000000000000000000.0]
        self.beta_on = True
        self.beta_decoder_on = beta_decoder_on
        self.betad_status = np.zeros((6, 1))

    def update_frequency_channels(self, channel: int):
        self.ch = channel - 1

    def update_deltad_settings(self, on_off: bool, threshold: int, delta_band: list):
        self.delta_decoder_on = on_off
        self.delta_th = threshold
        self.delta_band = delta_band

    def get_deltad_status(self):
        self.deltad_status[0, 0] = self.ch + 1
        self.deltad_status[1, 0] = self.delta_th
        self.deltad_status[2, 0] = self.delta_band[0]
        self.deltad_status[3, 0] = self.delta_band[1]
        self.deltad_status[4, 0] = self.delta_decoder_on
        self.deltad_status[5, 0] = self.delta_on
        # print(self.deltad_status[5, 0])

        return self.deltad_status

    def update_betad_settings(self, on_off: bool, threshold: float, beta_band: list):
        self.beta_decoder_on = on_off
        self.beta_th = threshold
        self.beta_band = beta_band

    def get_betad_status(self):
        self.betad_status[0, 0] = self.ch + 1
        self.betad_status[1, 0] = self.beta_th
        self.betad_status[2, 0] = self.beta_band[0]
        self.betad_status[3, 0] = self.beta_band[1]
        self.betad_status[4, 0] = self.beta_decoder_on
        self.betad_status[5, 0] = int(self.beta_on)

        return self.betad_status

    def process(self, seq: int, eeg_signal: np.ndarray) -> (int, np.ndarray, int, bool):

        # select the desired channel(s):
        eeg_signal = eeg_signal[self.ch, :]
        eeg_signal = np.reshape(eeg_signal, (1, eeg_signal.shape[0]))

        # accumulate the EEG signal and its sequence num in a buffer
        # if there are missing values, put NANs in the buffer
        seq_num_diff = fit_utils._signed_diff(self.previous_seq, seq, 16)
        if (self.previous_seq != -1) and (seq_num_diff > 1):  # connection issues
            missing_samples = 8 * (seq_num_diff - 1)
            if missing_samples > self.buffer_size_samples:
                missing_samples = self.buffer_size_samples
            # add NANs and the actual EEG block
            self.buffer_eeg.add(np.ones((self.num_channels, int(missing_samples))) * np.nan)
            # add NANs and store as the lost sequence numbers
            self.buffer_seq.add(np.ones((1, int(missing_samples))) * np.nan)

        self.buffer_eeg.add(eeg_signal)
        self.buffer_seq.add(np.ones((1, eeg_signal.shape[1])) * seq)

        # update seq num
        self.previous_seq = seq

        updated = False
        # only go further if the buffer is full
        if self.buffer_eeg.num_samples < self.buffer_size_samples:
            return self.delta, self.delta_on, self.beta, self.beta_on, seq, updated

        # the buffer is full
        data_loss = sum(sum(np.isnan(self.buffer_eeg.buffer)))

        if data_loss == 0:  # classification can be done only if there has been no data loss

            self.blocks_counter = self.blocks_counter + 1

            if self.blocks_counter >= self.update_blocks:

                self.blocks_counter = 0

                psd, freq = compute_welch_psd(eeg=self.buffer_eeg.buffer, window_type='hamming',
                                              window_size=self.srate, noverlap=int(self.srate / 2),
                                              nfft=4 * self.srate, sampling_rate=self.srate, channels=[0],
                                              max_freq=45, log10=False)
                self.delta_activity_check(psd, freq)
                self.beta_activity_check(psd, freq)

                if not self.beta_on:
                    self._waiting_sequence_num = seq
                    self._waiting = True

                updated = True

        if self._waiting:

            # If waiting beta inhibition

            # 1) Check current sequence number
            seq_num_diff2 = _signed_diff(self._waiting_sequence_num, seq, 16)
            # print('Waiting ' + str(seq_num_diff2 * self._decoder_calls_ms))

            # 2) If waiting time has been reached, then stop waiting and continue with next train of stimulations
            time_condition = seq_num_diff2 >= np.ceil(self.waiting_time / self.block_time)
            if time_condition and self._waiting:
                self._waiting = False
                self.beta_on = True

        return self.delta, self.delta_on, self.beta, self.beta_on, seq, updated

    def delta_activity_check(self, psd, freq):

        freq_pos = [(np.abs(freq - self.delta_band[x])).argmin() for x in range(len(self.delta_band))]
        delta_power = np.nanmean(psd[freq_pos[0]:freq_pos[1] + 1, :], axis=0)
        self.delta = delta_power

        if delta_power > self.delta_th and self.delta_decoder_on:
            # print('Delta 1')
            self.delta_on = True
        elif not self.delta_decoder_on:
            # print('Delta 2')
            self.delta_on = True
        else:
            # print('Delta 3')
            self.delta_on = False

    def beta_activity_check(self, psd, freq):

        freq_pos = [(np.abs(freq - self.beta_band[x])).argmin() for x in range(len(self.beta_band))]
        beta_power = np.nanmean(psd[freq_pos[0]:freq_pos[1] + 1, :], axis=0)
        self.beta = beta_power

        if beta_power < self.beta_th and self.beta_decoder_on:
            self.beta_on = True
        elif not self.beta_decoder_on:
            self.beta_on = True
        else:
            self.beta_on = False
