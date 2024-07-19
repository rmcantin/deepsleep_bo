import numpy as np
from scipy import signal
import random
from deps.eeg_utils import CircularBuffer
from bbt_pkg.fit.core.utils.fit_utils import _signed_diff
from auditory_stimulation_trigger import PLL
from typing import Tuple


class SlowWavesDecoder:

    """ Slow Waves Decoder based on PLL signal and amplitude thresholds to trigger Auditory Stimulation

    This class implements the decoder needed to detect slow waves and gather the information for the Auditory
    Stimulation module

    """

    def __init__(self, sampling_rate: int, num_channels: int, selected_channel: int,
                 filter_order: int, lower_cutoff_freq: float, upper_cutoff_freq: float,
                 k_pll: float, f_nco: float, target_phase: float, down_up_th: float, amplitude_th: Tuple,
                 volume: float, pll_timings: Tuple,
                 stim_probability: float):

        # General parameters
        self.sr = sampling_rate
        self.num_channels = num_channels  # Of the EEG signal
        self.selected_channel = (selected_channel - 1)  # Channel starting with 0
        self.decoder_calls_ms = 31.25  # Calls to the decoder module every 31.25 ms (hardware clock)

        # Audio parameters
        self.current_volume = volume

        # EEG signal filter parameters
        self.filter_order = filter_order
        self.lower_cutoff_freq = lower_cutoff_freq
        self.upper_cutoff_freq = upper_cutoff_freq
        freq_band = np.array([self.lower_cutoff_freq, self.upper_cutoff_freq])
        self.b_bp, self.a_bp = signal.butter(self.filter_order, freq_band * 2 / sampling_rate, 'bandpass')
        self.zi = np.zeros((self.num_channels, len(self.b_bp) - 1))

        # PLL - amplitude parameters
        self.amplitude_th = amplitude_th
        self.pll = PLL(k_pll=k_pll, f_nco=f_nco, sampling_rate=self.sr, target_phase=target_phase,
                       amplitude_th=self.amplitude_th, max_time_waiting_down_up=0.75, down_up_th=down_up_th)

        # SSD (Sleep scorer detector) state + Frequency detector state
        self.manual_mode_state = -1  # Deactivated by default
        self.auto_mode_state = 0
        self.stimulation_on = False
        self.stimulation_on_prev = False
        self.stimulation_triggered = False

        # Stimulation pattern parameters
        self.pll_on_time = pll_timings[0]  # ms
        self.pll_waiting_time = pll_timings[1]  # ms
        self.stim_probability = stim_probability  # probability of generating a stim. (instead of sham stim.)
        self.waiting_for_pll_on = True

        # Initialize EEG buffer
        self.buffer_size = 2 * self.sr
        self.buffer_eeg = CircularBuffer(self.num_channels, self.buffer_size)
        self.buffer_sequence = CircularBuffer(1, self.buffer_size)  # Store the EEG sequence numbers
        
        # Initialize sequence number parameters
        self.previous_sequence_num = -1  # Used to deal with connection losses
        self.waiting_sequence_num = -1  # Used to turn the pll on and off

        # Initialize output dictionaries
        self.decoder_status = np.zeros((16, 1))  # For log purposes
        self.detection = np.ones((9, 1))

    def update_stimulation_on_off(self, current_on_off_state: int, auto_mode: bool):

        # Update auto/manual N REM state value
        if auto_mode:
            self.auto_mode_state = current_on_off_state
        else:
            self.manual_mode_state = current_on_off_state

        # Set the stimulation ON OFF
        if self.manual_mode_state == -1 and self.auto_mode_state == 1:
            self.stimulation_on = True
        elif self.manual_mode_state == 1:
            self.stimulation_on = True
        else:
            self.stimulation_on = False

        if self.stimulation_on != self.stimulation_on_prev and self.stimulation_on:
            self.waiting_for_pll_on = True
            self.waiting_sequence_num = -1

        self.stimulation_on_prev = self.stimulation_on
        # print(self.stimulation_on_prev, self.stimulation_on)

    def update_filter_settings(self, filter_order: int, lower_cutoff_freq: float, upper_cutoff_freq: float):
        self.lower_cutoff_freq = lower_cutoff_freq
        self.upper_cutoff_freq = upper_cutoff_freq
        self.filter_order = filter_order
        self.b_bp, self.a_bp = signal.butter(self.filter_order, np.array([self.lower_cutoff_freq, self.upper_cutoff_freq]) * 2 / self.sr, 'bandpass')
        self.zi = np.zeros((self.num_channels, len(self.b_bp) - 1))

    def update_amplitude_thresholds(self, amplitude_th_min, amplitude_th_max, down_up_th):
        self.amplitude_th = (amplitude_th_min, amplitude_th_max)
        self.pll.amplitude_th = self.amplitude_th
        self.pll.set_down_up_th(down_up_th)

    def update_pll_settings(self, k_pll, f_nco, target_phase):
        self.pll.set_k_pll(k_pll)
        self.pll.set_f_nco(f_nco)
        self.pll.set_target_phase(target_phase)

    def update_sw_decoder_settings(self, channel: int, pll_on_time_ms: int, waiting_time_ms: int):
        self.selected_channel = int(channel-1)
        self.pll_waiting_time = waiting_time_ms
        self.pll_on_time = pll_on_time_ms

    def update_stimulation_volume(self, volume: float):
        self.current_volume = volume
    
    def get_decoder_status(self):
        self.decoder_status[0, 0] = self.manual_mode_state  # NREM state (manually)
        self.decoder_status[1, 0] = self.current_volume  # stimulation_volume
        self.decoder_status[2, 0] = self.amplitude_th[0]  # min amplitude threshold
        self.decoder_status[3, 0] = self.amplitude_th[1]  # max amplitude threshold
        # decoder_settings
        self.decoder_status[4, 0] = self.selected_channel+1
        self.decoder_status[5, 0] = self.pll_waiting_time
        # filter_settings
        self.decoder_status[6, 0] = self.filter_order
        self.decoder_status[7, 0] = self.lower_cutoff_freq
        self.decoder_status[8, 0] = self.upper_cutoff_freq
        # pll_settings
        self.decoder_status[9, 0] = self.pll.k_pll
        self.decoder_status[10, 0] = self.pll.f_nco
        self.decoder_status[11, 0] = self.pll.target_phase
        # Stimulation mode (ON/OFF)
        self.decoder_status[12, 0] = self.stimulation_on
        # pll_on_tim
        self.decoder_status[13, 0] = self.pll_on_time
        self.decoder_status[14, 0] = self.pll.down_up_th
        self.decoder_status[15, 0] = not self.waiting_for_pll_on

        return self.decoder_status
    
    def get_detection_status(self, stimulate, volume, th_min, th_max, channel, block_number, stim_sample,
                             down_up_before_stim, minimum_before_stim_sample):
        self.detection[0, 0] = stimulate
        self.detection[1, 0] = volume
        self.detection[2, 0] = th_min
        self.detection[3, 0] = th_max
        self.detection[4, 0] = channel
        self.detection[5, 0] = block_number
        self.detection[6, 0] = stim_sample
        self.detection[7, 0] = down_up_before_stim
        self.detection[8, 0] = minimum_before_stim_sample

    def process(self, sequence_num: int, eeg_signal: np.ndarray, eeg_orig_time: int):

        self.stimulation_triggered = False
        ts_detection = 0

        # Filter the signal (keeping the coefficients from previous filtering)
        eeg_signal_f, self.zi = signal.lfilter(self.b_bp, self.a_bp, eeg_signal, zi=self.zi)

        # --- BUFFER ---
        # Accumulate the EEG signal and its sequence num in a buffer. If there are missing values,
        # put NANs in the buffer

        seq_num_diff = _signed_diff(self.previous_sequence_num, sequence_num, 16)
        if (self.previous_sequence_num != -1) and (seq_num_diff > 1):  # connection loss
            missing_samples = 8 * (seq_num_diff-1)
            if missing_samples > self.buffer_size:
                missing_samples = self.buffer_size
            # add NANs and the actual EEG block
            self.buffer_eeg.add(np.ones((self.num_channels, int(missing_samples)))*np.nan)
            # add NANs and store as the lost sequence numbers
            self.buffer_sequence.add(np.ones((1, int(missing_samples)))*np.nan)
        self.buffer_eeg.add(eeg_signal_f)
        self.buffer_sequence.add(np.ones((1, eeg_signal_f.shape[1]))*sequence_num)

        # Channel selection
        eeg_signal_f1 = self.buffer_eeg.buffer[self.selected_channel, :]
        eeg_signal_f1 = np.reshape(eeg_signal_f1, (1, eeg_signal_f1.shape[0]))

        pll_output = np.full((1, 8), -1)

        self.pll.target_phase_crossed = False

        if self.waiting_for_pll_on:

            # 1) Check current sequence number
            seq_num_diff_pll_off = _signed_diff(self.waiting_sequence_num, sequence_num, 16)
            # print('Esperando ' + str(float(np.ceil(seq_num_diff_pll_off)) / self.decoder_calls_ms))

            # 2) If waiting time has been reached, then stop waiting and turn pll on
            time_condition = seq_num_diff_pll_off >= np.ceil(self.pll_waiting_time / self.decoder_calls_ms)
            if time_condition and self.waiting_for_pll_on:
                self.waiting_for_pll_on = False
                self.waiting_sequence_num = sequence_num

        else:

            # Check current sequence number
            seq_num_diff_pll_on = _signed_diff(self.waiting_sequence_num, sequence_num, 16)
            # print('Funcionando ' + str(float(np.ceil(seq_num_diff_pll_on)) / self.decoder_calls_ms))

            # If pll on time has been reached, then stop running the pll and start waiting time
            time_condition = seq_num_diff_pll_on >= np.ceil(self.pll_on_time / self.decoder_calls_ms)
            if time_condition and not self.waiting_for_pll_on:
                self.waiting_for_pll_on = True
                self.waiting_sequence_num = sequence_num

        # For each new sample update the pll - amplitude decoder and check if target phase is crossed
        for k in range(eeg_signal.shape[1]):

            pll_output[0, k] = self.pll.update(eeg_signal_f1[0, -eeg_signal.shape[1] + k], sequence_num * 8 + k)
            # pll_output[0, k] = self.pll.update(eeg_signal[4, -eeg_signal.shape[1] + k], sequence_num * 8 + k)
            self.pll.check_stimulation(None, sequence_num * 8 + k)
            if self.pll.target_phase_crossed:
            # if eeg_signal[4, k] > 0.78*50000 and eeg_signal[4, k+1] < 0.82*50000 and (eeg_signal[4, k]<eeg_signal[4, k+1]):
            # if eeg_signal[4, k] > -0.6*50000 and eeg_signal[4, k+1] < -0.56*50000 and (eeg_signal[4, k]<eeg_signal[4, k+1]):
            # if eeg_signal_f1[4, k] < 0.61*50000 and eeg_signal_f1[4, k+1] > 0.55*50000 and (eeg_signal_f1[4, k]>eeg_signal_f1[4, k+1]):
            # if eeg_signal_f[4, k] < 0 and eeg_signal_f[4, k+1] > 0 and (eeg_signal_f[4, k]<eeg_signal_f[4, k+1]):

                self.stimulation_triggered = True

                # True or sham stimulation
                p = self.stim_probability
                true_stim = random.choices([0, 1], weights=[1-p, p])
                if self.stimulation_on and true_stim[0] and not self.waiting_for_pll_on == 1:
                    stimulate = True
                    volume = self.current_volume
                elif self.stimulation_on and true_stim[0] and self.waiting_for_pll_on == 1:
                    stimulate = False
                    volume = 0
                elif self.stimulation_on and true_stim[0] == 0:  # Sham stimulation
                    stimulate = True
                    volume = 0
                else:
                    stimulate = False
                    volume = self.current_volume

                # Update detection status
                detection_sample = eeg_signal.shape[1]-k-1
                ts_detection = eeg_orig_time - detection_sample/self.sr * (10**6)
                self.get_detection_status(stimulate=stimulate, volume=volume, th_min=self.amplitude_th[0],
                                          th_max=self.amplitude_th[1], channel=self.selected_channel+1,
                                          block_number=sequence_num, stim_sample=detection_sample,
                                          down_up_before_stim=detection_sample + (self.pll.down_up_before_stim_pos
                                                                                  - self.pll.stimulation_pos),
                                          minimum_before_stim_sample=detection_sample + (self.pll.minimum_before_stim_pos -
                                                                     self.pll.stimulation_pos))

        self.previous_sequence_num = sequence_num

        return pll_output, self.detection, ts_detection, self.stimulation_triggered
