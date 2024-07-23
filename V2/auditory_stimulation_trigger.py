import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import warnings
from scipy.stats import circmean, circstd, sem


class PLL:

    def __init__(self, k_pll: float, f_nco: float, sampling_rate: int, target_phase: float, amplitude_th: Tuple,
                 max_time_waiting_down_up: float, down_up_th: float):

        # PLL parameters
        self.k_pll = k_pll
        self.f_nco = f_nco
        self.target_phase = target_phase
        self.sampling_rate = sampling_rate

        # Amplitude parameters
        self.amplitude_th = amplitude_th
        self.down_up_th = down_up_th
        self.max_time_waiting_down_up = max_time_waiting_down_up

        # PLL execution variables initialization
        self.phase_error = 0.0
        self.phase = 0.0
        self.phase_prev = target_phase + 1
        self.frequency = 0.0
        self.stimulation_phase = 0
        self.stimulation_pos = -1
        self.down_up_before_stim_pos = -1
        self.minimum_before_stim_pos = -1
        self.target_phase_crossed = False

        self.phase_error_integral = 0

        # Amplitude thresholding variables initialization
        self.up_down_found = False
        self.up_down_pos = 0
        self.so_buffer = []
        self.down_up_found = False
        self.down_up_pos = 0
        self.so_found = False
        self.pos = 0
        self.prev_sample = 0.0
        self.min_value = 0
        self.min_value_pos = 0

    def set_k_pll(self, k_pll):
        self.k_pll = k_pll

    def set_f_nco(self, f_nco):
        self.f_nco = f_nco

    def set_target_phase(self, target_phase):
        self.target_phase = target_phase

    def set_down_up_th(self, down_up_th):
        self.down_up_th = down_up_th

    def update(self, input_signal, pos):

        self.target_phase_crossed = False
        self.pos = pos

        # --- Amplitude ---

        # 1) Check if too much time waiting for a down up crossing
        if (self.pos - self.up_down_pos) / self.sampling_rate > self.max_time_waiting_down_up and not(self.down_up_found) and self.up_down_found:
            self.up_down_found = False
            self.down_up_found = False
            self.so_found = False
            self.so_buffer = []

        # 2) Find up down crossing
        if self.prev_sample > 0 > input_signal:
            self.up_down_found = True
            self.up_down_pos = self.pos
            self.so_buffer = []

        # 3) Buffer the SO if up down crossing found
        if self.up_down_found:
            self.so_buffer.append(input_signal)

        # 4) If down up crossing is found after up down crossing, save key points
        if self.prev_sample < self.down_up_th < input_signal and self.up_down_found:
            self.down_up_found = True
            self.down_up_pos = self.pos
            self.min_value = np.nanmin(self.so_buffer)
            self.min_value_pos = self.up_down_pos + np.where(self.so_buffer == np.nanmin(self.so_buffer))[0][0] - 1
            if not np.any(self.so_buffer == np.nan):
                if self.amplitude_th is not None:
                    if self.amplitude_th[0] < self.min_value < self.amplitude_th[1]:
                        self.so_found = True
                    else:
                        self.so_found = False
                else:
                    self.so_found = True
            else:
                self.so_found = False
                

        # 5) Reset SO parameters if needed
        if self.down_up_found:
            self.up_down_found = False
            self.down_up_found = False
            self.so_buffer = []

        self.prev_sample = input_signal

        # --- PLL ---

        # Phase detector
        phase_detector_output = np.cos(self.phase)

        # NCO control signal
        self.phase_error = input_signal * phase_detector_output

        # NCO frequency and phase update
        self.frequency = self.f_nco + self.k_pll * self.phase_error
        self.phase_prev = self.phase
        self.phase = self.phase + 2 * np.pi * self.frequency / self.sampling_rate

        # NCO output (cosine wave)
        pll_output = 50 * np.cos(self.phase)
        
        """
        # Phase detector
        self.phase_error = np.angle(np.exp(1j * (input_signal - self.phase)))

        # Update phase error integral
        self.phase_error_integral += self.phase_error

        # Calculate control voltage (simple PI controller)
        loop_bandwidth = self.k_pll
        loop_damping = 1
        control_voltage = loop_bandwidth * self.phase_error + loop_bandwidth * loop_damping * self.phase_error_integral

        # Update phase and frequency
        self.phase_prev = self.phase
        self.phase += 2 * np.pi * (self.f_nco + control_voltage) / self.sampling_rate
        self.f_nco += control_voltage / (2 * np.pi * self.sampling_rate)

        pll_output = np.sin(self.phase)
        """

        return pll_output

    def check_stimulation(self, eeg_phase, time_pos):

        # If PLL crosses target phase, save stimulation values
        # if np.rad2deg(self.phase_prev) % 360 < self.target_phase < np.rad2deg(self.phase) % 360:
        theta1 = ((np.rad2deg(self.phase_prev) - self.target_phase) + 180) % 360 - 180
        theta2 = ((np.rad2deg(self.phase) - self.target_phase) + 180) % 360 - 180
  
        if theta1 < 0 and theta2 > 0:
        # if (self.target_phase - np.rad2deg(self.phase_prev)) % 360 > 0 and \
        #         (np.rad2deg(self.phase) - np.rad2deg(self.target_phase)) % 360 > 0:

            # print(np.rad2deg(self.phase_prev) % 360, self.target_phase, np.rad2deg(self.phase) % 360)
            # print((self.target_phase - np.rad2deg(self.phase_prev)) % 360,
            #       (np.rad2deg(self.phase) - self.target_phase) % 360, self.so_found)
            
            if self.so_found:
                self.target_phase_crossed = True
                self.stimulation_phase = eeg_phase
                self.stimulation_pos = time_pos
                self.down_up_before_stim_pos = self.down_up_pos
                self.minimum_before_stim_pos = self.min_value_pos
                self.so_found = False
                #print(self.target_phase_crossed, np.rad2deg(self.phase_prev) % 360, np.rad2deg(self.phase) % 360)


class AuditoryStimulationEvaluation:

    def __init__(self, stimulation_phase, stimulation_position,  down_up_before_stim_pos, minimum_before_stim_pos,
                 output_path):
        self.stimulation_phase = stimulation_phase
        self.stimulation_position = stimulation_position
        self.down_up_before_stim_pos = down_up_before_stim_pos
        self.minimum_before_stim_pos = minimum_before_stim_pos
        self.cmae = None
        self.cm = None
        self.csd = None
        self.pas_up_phase = None
        self.pas_not_up_phase = None
        self.p_up_phase_stimulated = None
        self.output_path = output_path

    def phase_tracking_metrics(self, target_phase):

        self.cmae = np.abs(np.rad2deg(circmean(self.stimulation_phase, nan_policy='omit')) - target_phase) / 180
        self.cm = np.rad2deg(circmean(self.stimulation_phase, nan_policy='omit'))
        self.csd = np.rad2deg(circstd(self.stimulation_phase, nan_policy='omit'))

    def _percentage_active_stimulations(self, target_range: Tuple, n_nrem_win: int, win_length: float, max_freq: float):
        num_stim_target_phase = len([x for x in self.stimulation_phase if (np.deg2rad(target_range[0]) < x <
                                                                           np.deg2rad(target_range[1]))])
        max_stim = max_freq * win_length
        pas = num_stim_target_phase / (n_nrem_win * max_stim)

        return pas

    def percentage_stimulations_up_phase(self, target_range: Tuple = (0, 90)):
        num_stim_target_phase = len([x for x in self.stimulation_phase if (np.deg2rad(target_range[0]) < x <
                                                                           np.deg2rad(target_range[1]))])
        print('Number of stimulations in the up phase: ', num_stim_target_phase)
        self.p_up_phase_stimulated = (num_stim_target_phase / len(self.stimulation_phase)) * 100

    def percentage_active_stimulations_up_phase(self, up_phase: Tuple = (0, 90), n_nrem_win: int = None,
                                                win_length: float = 2, max_freq_stim: float = 2):

        self.pas_up_phase = self._percentage_active_stimulations(up_phase, n_nrem_win, win_length, max_freq_stim)

    def percentage_active_stimulations_not_up_phase(self, up_phase: Tuple = (0, 90), n_nrem_win: int = None,
                                                    win_length: float = 2, max_freq_stim:float = 2):

        if self.pas_up_phase is None:
            self.pas_up_phase = self._percentage_active_stimulations(up_phase, n_nrem_win, win_length, max_freq_stim)

        pas_all = self._percentage_active_stimulations((-1, 361), n_nrem_win, win_length, max_freq_stim)
        self.pas_not_up_phase = pas_all - self.pas_up_phase

    def ecludian_distance_single_recording(self):

        minimum = np.nanmin([self.cmae, self.pas_not_up_phase, self.pas_up_phase])
        maximum = np.nanmax([self.cmae, self.pas_not_up_phase, self.pas_up_phase])
        cmae_norm = (self.cmae - minimum) / (maximum - minimum)
        pas_not_up_phase_norm = (self.pas_not_up_phase - minimum) / (maximum - minimum)
        pas_up_phase_norm = (self.pas_up_phase - minimum) / (maximum - minimum)
        self.ed = np.sqrt(cmae_norm**2 + pas_not_up_phase_norm**2 + (1 - pas_up_phase_norm)**2)

    def plot_phase_distribution(self):

        fig, axs = plt.subplots(subplot_kw=dict(projection='polar'))

        axs.hist(self.stimulation_phase, bins=np.linspace(0, 2 * np.pi, 72), ec='black')
        axs.axvline(circmean(self.stimulation_phase, nan_policy='omit'), color='r', linewidth=2)
        axs.axvline(circmean(self.stimulation_phase, nan_policy='omit') +
                          circstd(self.stimulation_phase, nan_policy='omit'),
                          color='lime', linewidth=2)
        axs.axvline(circmean(self.stimulation_phase, nan_policy='omit') -
                          circstd(self.stimulation_phase, nan_policy='omit'),
                          color='lime', linewidth=2)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(self.output_path + '_phase_distribution.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_mean_sem_segments(self, signal, segment_interval: Tuple, point: str):

        segments = []

        if point == 'stim':
            points = self.stimulation_position
        elif point == 'down_up':
            points = self.down_up_before_stim_pos
        elif point == 'minimum':
            points = self.minimum_before_stim_pos
        else:
            warnings.warn('Only stim, down_up or minimum values allowed. Stim selected as default')
            points = self.stimulation_position

        for start in points:
            start_idx = start - segment_interval[0]
            end_idx = start + segment_interval[1]
            if start_idx >= 0:
                segment = signal[start_idx:end_idx]
                if len(segment) != (segment_interval[0] + segment_interval[1]):
                    segment = np.concatenate(
                        (signal[start_idx:], np.repeat(np.nan, int(segment_interval[0] + segment_interval[1] -
                                                                   len(signal[start_idx:])))))
                segments.append(segment)

        mean_values = np.nanmean(segments, axis=0)
        sem_values = sem(segments, axis=0, nan_policy='omit')

        plt.figure(figsize=(10, 6))
        plt.plot(mean_values, label='Mean')
        plt.fill_between(range(len(mean_values)), mean_values - sem_values, mean_values + sem_values, alpha=0.3,
                         label='Standard Error')
        plt.axvline(segment_interval[0], color='k')
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.title('Mean and Standard Error across Segments')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(self.output_path + '_mean_sem_segments_' + point + '.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_phase_morphology(self):
        pass
