import os
import pickle
from collections import Counter
import numpy as np
from auditory_stimulation_trigger import AuditoryStimulationEvaluation
from frequencies_detector import FrequenciesDetector
from slow_waves_decoder import SlowWavesDecoder
import math
from scipy.signal import butter, hilbert, lfilter


def load_fit_and_save_temporal(file):

    py_dict = pickle.load(open(file, "rb"))
    properties = py_dict['properties']
    signals = py_dict['signals']
    events = py_dict['events']
    log = py_dict['log']
    parameters = py_dict['parameters']

    return properties, signals, events, log, parameters


def load_info(signal, events, log_t0, fs_st):
    # Channel used for slow wave decoder
    sw_channel = int(Counter(events['detection'].values[4]).most_common(1)[0][0])
    # Manual mode
    ssd_value_changes = np.rint(1e-6 * (events['ssd_value'].ts - log_t0) * fs_st).astype(int)
    ssd_values = events['ssd_value'].values

    manual_mode_changes_on = np.where(ssd_values == 1)[1]
    manual_mode_changes_off = np.where(ssd_values == 0)[1]
    ss_manual_on = []
    ss_manual_off = []

    # Manual ON
    try:
        ss_manual_on.append([ssd_value_changes[manual_mode_changes_on], ssd_value_changes[manual_mode_changes_on + 1]])

    except:
        manual_mode_changes_on2 = manual_mode_changes_on[0: len(manual_mode_changes_on) - 1]
        ss_manual_on.append([ssd_value_changes[manual_mode_changes_on2], ssd_value_changes[manual_mode_changes_on2 + 1]])
        ss_manual_on[0][0] = np.append(ss_manual_on[0][0], ssd_value_changes[-1])
        ss_manual_on[0][1] = np.append(ss_manual_on[0][1], signal.shape[1])

    ss_manual_on = np.stack(ss_manual_on[0], axis=1)

    # Manual OFF
    try:
        ss_manual_off.append([ssd_value_changes[manual_mode_changes_off], ssd_value_changes[manual_mode_changes_off + 1]])

    except:
        manual_mode_changes_off2 = manual_mode_changes_off[0: len(manual_mode_changes_off) - 1]
        ss_manual_off.append([ssd_value_changes[manual_mode_changes_off2], ssd_value_changes[manual_mode_changes_off2 + 1]])
        ss_manual_off[0][0] = np.append(ss_manual_off[0][0], ssd_value_changes[-1])
        ss_manual_off[0][1] = np.append(ss_manual_off[0][1], signal.shape[1])

    ss_manual_off = np.stack(ss_manual_off[0], axis=1)

    stimulation_on_ssd = events['ssd_prediction'].values

    # Change positions where manual mode was used for sleep stage decoding
    stimulation_on_ssd_annot = stimulation_on_ssd.copy()

    for i in range(ss_manual_on.shape[0]):
        # Manual mode = 1 -> stimulation  on
        stimulation_on_ssd_annot[0, ss_manual_on[i, 0]: ss_manual_on[i, 1]] = 2

    for i in range(ss_manual_off.shape[0]):
        # Manual mode = 0 -> stimulation off
        stimulation_on_ssd_annot[0, ss_manual_off[i, 0]: ss_manual_off[i, 1]] = 3

    return sw_channel, stimulation_on_ssd_annot


def decoders_process(info_ssd, channel, fs, signal, seq_eeg, ts_eeg, volume, freq_power_buffer_size,                # seq_eeg and ts_eeg are new, stim_probability is missing
                     lower_cutoff_freq, upper_cutoff_freq, filter_order, delta_band, beta_band, pll_on_time,
                     waiting_time, delta_power_th, beta_power_th, beta_change_th, threshold_min, threshold_max,     # beta_change_th is new
                     down_up_th, k_pll, f_nco, target_phase):
    # Sleep stage decoder
    stim_on_ssd = info_ssd[0, :]
    
    manual_off = np.where(stim_on_ssd == 3)[0]  # These 4 lines are new
    manual_on = np.where(stim_on_ssd == 2)[0]   #
    stim_on_ssd[manual_off] = 0                 #
    stim_on_ssd[manual_on] = 1                  #

    # Initialize frequency detector
    fd = FrequenciesDetector(selected_channel=channel, sampling_rate=fs, delta_band=delta_band,
                             delta_th=delta_power_th, beta_band=beta_band, beta_th=beta_power_th,
                             beta_change_th=beta_change_th, buffer_size=freq_power_buffer_size, delta_decoder_on=False,     # beta_change_th is new, delta_decoder_on is now False and beta_decoder_on is now False
                             beta_decoder_on=False)
    delta_power = np.repeat(-np.inf, signal.shape[1])
    stim_on_delta = np.zeros((signal.shape[1],))
    beta_power = np.repeat(-np.inf, signal.shape[1])
    stim_on_beta = np.ones((signal.shape[1],))

    # Initialize slow waves decoder
    sw_decoder = SlowWavesDecoder(selected_channel=channel, num_channels=signal.shape[0],
                                  sampling_rate=fs, amplitude_th=(threshold_min, threshold_max),
                                  volume=volume, pll_timings=(pll_on_time, waiting_time),
                                  filter_order=filter_order,
                                  lower_cutoff_freq=lower_cutoff_freq, upper_cutoff_freq=upper_cutoff_freq, k_pll=k_pll,    # stim_probability is missing
                                  f_nco=f_nco, target_phase=target_phase, down_up_th=down_up_th)
    stim_on_total = np.zeros((signal.shape[1],))
    stimulate = []
    stims_volume = []   # This is new
    detection_seq = []
    detection_samples = []
    down_up_before_stim = []
    minimum_before_stim = []

    # ...Iterate...
    step = 8
    epoch_length = step
    num_epochs = math.floor((signal.shape[1] - epoch_length) / step)
    
    # seq_eeg = np.array(range(1, num_epochs + 2))          These two lines were removed since the variables
    # ts_eeg = np.zeros((num_epochs + 1,))                  are now given as parameters of the function

    for epoch_i in range(num_epochs + 1):
        epoch_ini = epoch_i * step
        epoch_end = epoch_ini + epoch_length

        if sum(sum(np.isnan(signal[:, epoch_ini:epoch_end]))) == 0:
            # ------------- Sleep stage decoder -------------
            
            ssd_on = int(Counter(stim_on_ssd[epoch_ini:epoch_end]).most_common(1)[0][0])
            if ssd_on == -1:
                ssd_on = 0
            # ------------- Frequencies detector -------------
            delta, delta_on, beta, beta_on, seq_freq, updated_freq = fd.process(seq_eeg[epoch_i],
                                                                                signal[:, epoch_ini:epoch_end])
            delta_power[epoch_ini:epoch_end] = delta[0]
            if delta_on:
                stim_on_delta[epoch_ini:epoch_end] = 1
            beta_power[epoch_ini:epoch_end] = beta[0]
            if not beta_on:
                stim_on_beta[epoch_ini:epoch_end] = 0

            # ------------- Slow waves decoder -------------
            # Stimulate ?
            if ssd_on and delta_on and beta_on:                                                             # The # Stimulate? segment now starts before the next segment
                auto_mode_value = 1                                                                         #
                stim_on_total[epoch_ini:epoch_end] = 1                                                      #
            else:                                                                                           #
                auto_mode_value = 0                                                                         #
            sw_decoder.update_stimulation_on_off(auto_mode_value, auto_mode=True)                           #
            
            # These following lines were above the segment that starts with # Stimulate?
            pll_output, detection_values, _, sw_updated = \
                sw_decoder.process(seq_eeg[epoch_i], signal[:, epoch_ini:epoch_end], ts_eeg[epoch_i])       #
                                                                                                            #
            if sw_updated:                                                                                  #
                stimulate.append(sw_decoder.detection[0, 0])                                                #
                stims_volume.append(sw_decoder.detection[1, 0])                                             # stims_volume is new
                detection_seq.append(sw_decoder.detection[5, 0])                                            #
                detection_samples.append(sw_decoder.detection[6, 0])                                        #
                down_up_before_stim.append(sw_decoder.detection[7, 0])                                      #
                minimum_before_stim.append(sw_decoder.detection[8, 0])                                      #

    ini_seq_num = np.nonzero(np.in1d(seq_eeg, np.asarray(detection_seq)))[0] * 8
    end_seq_num = ini_seq_num + 7

    stimulate_enabled_idx = np.where(np.logical_and(np.asarray(stimulate) == 1, np.asarray(stims_volume) != 0))[0]          # This didn't include the logical_and with stims_volume
    stimulus_positions = (end_seq_num[stimulate_enabled_idx] - np.asarray(detection_samples)[stimulate_enabled_idx]) \
                         + 100 * 1e-3 * fs
    stimulus_positions = np.rint(stimulus_positions).astype(int)
    down_up_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(down_up_before_stim)[stimulate_enabled_idx]).astype(int)
    minimum_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(minimum_before_stim)[stimulate_enabled_idx]).astype(int)

    n_detections = len(detection_samples)               # These two variables were global
    n_stimulations = len(stimulus_positions)            #
    print('Number of detections: ', n_detections)
    print('Number of stimulations: ', n_stimulations)

    # Events
    stim_on_total = np.zeros((1, stim_on_ssd.shape[0]))
    
    stim_on_total[0, np.where(np.count_nonzero(np.vstack([stim_on_ssd, stim_on_delta, stim_on_beta]) == 1,
                                               axis=0) == 3)[0]] = 1
    stim_on_off = np.vstack((stim_on_ssd, stim_on_delta, stim_on_beta, stim_on_total))

    time_stim_auto_gates_on = len(np.where(stim_on_total == 1)[0]) / fs

    return stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, \
        time_stim_auto_gates_on, n_detections, n_stimulations                                                       # The function now returns n_detections and n_stimulations aswell


def decoders_process_nrem_window(channel, fs, signal, seq_eeg, ts_eeg, volume, lower_cutoff_freq, upper_cutoff_freq,
                                 filter_order, pll_on_time, waiting_time, threshold_min, threshold_max, down_up_th,
                                 k_pll, f_nco, target_phase):

    stim_on_ssd = np.ones((signal.shape[1],))
    stim_on_beta = np.ones((signal.shape[1],))
    stim_on_delta = np.ones((signal.shape[1],))
    # Initialize slow waves decoder
    sw_decoder = SlowWavesDecoder(selected_channel=channel, num_channels=signal.shape[0],
                                  sampling_rate=fs, amplitude_th=(threshold_min, threshold_max),
                                  volume=volume, pll_timings=(pll_on_time, waiting_time),
                                  filter_order=filter_order,
                                  lower_cutoff_freq=lower_cutoff_freq, upper_cutoff_freq=upper_cutoff_freq, k_pll=k_pll,
                                  f_nco=f_nco, target_phase=target_phase, down_up_th=down_up_th)

    stimulate = []
    stims_volume = []
    detection_seq = []
    detection_samples = []
    down_up_before_stim = []
    minimum_before_stim = []

    # ...Iterate...
    step = 8
    epoch_length = step
    num_epochs = math.floor((signal.shape[1] - epoch_length) / step)

    for epoch_i in range(num_epochs + 1):
        epoch_ini = epoch_i * step
        epoch_end = epoch_ini + epoch_length

        if sum(sum(np.isnan(signal[:, epoch_ini:epoch_end]))) == 0:
            # ------------- Slow waves decoder -------------
            sw_decoder.update_stimulation_on_off(1, auto_mode=True)

            pll_output, detection_values, _, sw_updated = \
                sw_decoder.process(seq_eeg[epoch_i], signal[:, epoch_ini:epoch_end], ts_eeg[epoch_i])

            if sw_updated:
                stimulate.append(sw_decoder.detection[0, 0])
                stims_volume.append(sw_decoder.detection[1, 0])
                detection_seq.append(sw_decoder.detection[5, 0])
                detection_samples.append(sw_decoder.detection[6, 0])
                down_up_before_stim.append(sw_decoder.detection[7, 0])
                minimum_before_stim.append(sw_decoder.detection[8, 0])

    ini_seq_num = np.nonzero(np.in1d(seq_eeg, np.asarray(detection_seq)))[0] * 8
    end_seq_num = ini_seq_num + 7

    stimulate_enabled_idx = np.where(np.logical_and(np.asarray(stimulate) == 1, np.asarray(stims_volume) != 0))[0]
    stimulus_positions = (end_seq_num[stimulate_enabled_idx] - np.asarray(detection_samples)[stimulate_enabled_idx]) \
                         + 100 * 1e-3 * fs
    stimulus_positions = np.rint(stimulus_positions).astype(int)
    down_up_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(down_up_before_stim)[stimulate_enabled_idx]).astype(int)
    minimum_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(minimum_before_stim)[stimulate_enabled_idx]).astype(int)

    n_detections = len(detection_samples)
    n_stimulations = len(stimulus_positions)
    print('Number of detections: ', n_detections)
    print('Number of stimulations: ', n_stimulations)

    # Events
    stim_on_total = np.zeros((1, stim_on_ssd.shape[0]))
    stim_on_total[0, np.where(np.count_nonzero(np.vstack([stim_on_ssd, stim_on_delta, stim_on_beta]) == 1,
                                               axis=0) == 3)[0]] = 1
    stim_on_off = np.vstack((stim_on_ssd, stim_on_delta, stim_on_beta, stim_on_total))

    time_stim_auto_gates_on = len(np.where(stim_on_total == 1)[0]) / fs

    return stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, \
        time_stim_auto_gates_on, n_detections, n_stimulations


def filtering(signal, fs, lower_cutoff_freq, upper_cutoff_freq, order):
    """ Filter the signal """
    # 1. remove NaNs, 2. interpolation, 3. filter, 4. add NaNs again
    not_nan = np.logical_not(np.isnan(signal))
    # Save NaN indices to put them back into the signal later on
    missing_values_mask = np.invert(not_nan)
    nan_idx = np.argwhere(missing_values_mask)

    indices = np.arange(signal.shape[0])
    signal_interp = np.interp(indices, indices[not_nan], signal[not_nan])
    b_bp, a_bp = butter(order, np.array([lower_cutoff_freq, upper_cutoff_freq]) * 2 / fs, 'bandpass')
    signal_f = lfilter(b_bp, a_bp, signal_interp)
    signal_f[nan_idx] = np.nan

    return signal_f

# This comes from the old optimizer.py, I added it myself for treating the old data
def resample_binarize_labels(y, fs):
    N2 = np.where(y == 2)[0]
    N3 = np.where(y == 3)[0]
    N2_N3 = np.concatenate((N2, N3))
    indices = np.arange(0, len(y))
    mask_1 = np.isin(indices, N2_N3)
    mask_0 = np.isin(indices, N2_N3, invert=True)
    y_binarized = y.copy()
    y_binarized[mask_1], y_binarized[mask_0] = 1, 0
    y_resampled_binarized = np.repeat(y_binarized, fs * 30)
    y_resampled_binarized = np.reshape(y_resampled_binarized, (1, y_resampled_binarized.shape[0]))
    return y_resampled_binarized


def visualize_results(filtered_signal, target_phase, stimulus_samples_auto, stimulation_on_off, fs,
                      time_stim_on_total, down_up_before_stim_pos, minimum_before_stim_pos):
    stimulation_on_ssd = stimulation_on_off[0, :]
    stimulation_on_delta = stimulation_on_off[1, :]
    stimulation_on_beta = stimulation_on_off[2, :]

    print('----- Auto Mode -----')
    print(f'Stimulation ON (sleep stage decoder): {len(np.where(stimulation_on_ssd == 1)[0]) / fs}s')
    print(f'Stimulation ON (delta decoder):  {len(np.where(stimulation_on_delta == 1)[0]) / fs}s')
    print(f'Stimulation ON (beta decoder): {len(np.where(stimulation_on_beta == 1)[0]) / fs}s')
    print(f'Stimulation ON total: {time_stim_on_total}s')

    # Obtain phase of the signal

    # Compute analytic signal:
    # 1. remove NaNs, 2. interpolation, 3. add NaNs again

    not_nan = np.logical_not(np.isnan(filtered_signal))
    missing_values_mask = np.invert(not_nan)
    nan_idx = np.argwhere(missing_values_mask)
    indices = np.arange(filtered_signal.shape[0])
    signal_metrics_interp = np.interp(indices, indices[not_nan], filtered_signal[not_nan])
    phase_eeg = (np.angle(hilbert(signal_metrics_interp), deg=False) + np.pi / 2) % (2 * np.pi)
    phase_eeg_round = np.round(phase_eeg, 3)
    count = Counter(phase_eeg_round)
    phase_eeg[nan_idx] = np.nan
    if len(count) != 6284:
        # 1. remove outliers, 2. interpolation, 3. filter
        p_signal_95 = np.percentile(np.abs(filtered_signal[not_nan]), 95)
        inliers = np.where(np.abs(filtered_signal) <= p_signal_95)[0]
        signal_metrics_interp = np.interp(indices, indices[inliers], filtered_signal[inliers])
        phase_eeg = (np.angle(hilbert(signal_metrics_interp), deg=False) + np.pi / 2) % (2 * np.pi)
        phase_eeg[nan_idx] = np.nan

    # Obtain metrics
    ase = AuditoryStimulationEvaluation(phase_eeg[stimulus_samples_auto], stimulus_samples_auto,
                                        down_up_before_stim_pos, minimum_before_stim_pos, None)

    ase.phase_tracking_metrics(target_phase)
    ase.percentage_active_stimulations_up_phase(n_nrem_win=time_stim_on_total / 2)
    ase.percentage_active_stimulations_not_up_phase(n_nrem_win=time_stim_on_total / 2)
    ase.ecludian_distance_single_recording()
    print(f'cm: {ase.cm}, csd: {ase.csd}, cmae: {ase.cmae}, pas_up_phase: {ase.pas_up_phase},'
          f' pas_not_up_phase: {ase.pas_not_up_phase}')

    # Create phase plot, average waveform time-locked to stim, down up zc, and minimum plots
    # ase.plot_phase_distribution()
    # ase.plot_mean_sem_segments(filtered_signal, (int(1.5 * fs), fs), point='stim')

    return ase.cm, ase.csd, ase.cmae, ase.pas_up_phase, ase.pas_not_up_phase


def process(dir, file, threshold_max, k_pll, f_nco, start_sample=None, end_sample=None):
    is_old_data = False

    for folder in os.listdir(dir):
        if file in folder:
            folder_name = folder
            folder_path = os.path.join(dir, folder_name)
            for file in os.listdir(folder_path):
                if "_ST" in file and file.endswith('.npz'):
                    file_name = file
                    is_old_data = True
                    break
                elif file.endswith('.p'):
                    file_name = file
                    break
            break
            
    file_path = os.path.join(folder_path, file_name)

    # ---------------------------------- Load & filtering ----------------------------------

    # Load npz file
    if is_old_data:
        data = np.load(file_path, allow_pickle=True)
        print("LOADING FILE:",file_path)
    else:
        prop, signals, events, log, parameters = load_fit_and_save_temporal(file_path)

    # Load signal, fs, and labels
    if is_old_data:
        signal = data['x'].transpose(2, 0, 1).reshape(data['x'].shape[2], -1)
        fs = int(data['fs'][0])
        gt_labels = data['y']
        gt_labels_annot = resample_binarize_labels(gt_labels, fs)
        
        if '_ST2_' in file_name:  # No HB data after 497 min
            signal = signal[:, 0:497*60*fs]
            gt_labels_annot = gt_labels_annot[:, 0:497*60*fs]
        info_ssd = gt_labels_annot
    
    if is_old_data:
        step = 8                                                            # These lines where part of the decoders_process function,
        epoch_length = step                                                 # but they are constant so we canpre-calculate the values
        num_epochs = math.floor((signal.shape[1] - epoch_length) / step)    # since they are now parameters we need to give to the function.
        seq_eeg = np.array(range(1, num_epochs + 2))                        #
        ts_eeg = np.zeros((num_epochs + 1,))                                # With the params above we can now calculate seq_eeg and ts_eeg
    else:
        # Find signal id
        signal_type = 'eeg'
        prop_eeg = dict(filter(lambda item: item[1].signal_type == signal_type, prop.signals.items()))
        if len(prop_eeg.keys()) == 0:
            print('signal type not found {}'.format(signal_type))
            exit(1)

        signal_id = next(iter(prop_eeg.keys()))

        # Load signal
        signal = signals[signal_id].values
        fs = prop.signals[signal_id].sampling_rate
        fs_st = log.signals[signal_id].sr_st
        log_t0 = log.signals[signal_id].t0
        seq_eeg = signals[signal_id].seq
        ts_eeg = signals[signal_id].ts

        # Load information from events
        channel, gt_labels = load_info(signal, events, log_t0, fs_st)
        info_ssd = gt_labels

    # Filter signal
    if is_old_data:
        lower_cutoff_freq = 0.2
        upper_cutoff_freq = 5
        filter_order = 2
    else:
        lower_cutoff_freq = float(parameters['DecoderMediator']['lower_cutoff_freq']['value'])
        upper_cutoff_freq = float(parameters['DecoderMediator']['upper_cutoff_freq']['value'])
        filter_order = int(parameters['DecoderMediator']['filter_order']['value'])

    # ---------------------------------- Define parameters ----------------------------------
    if is_old_data:
        volume = 0.1
        channel = 10
    else:
        volume = float(parameters['DecoderMediator']['volume']['value'])

    # Delta and beta
    if is_old_data:
        freq_power_buffer_size = 4
        delta_band = [0.5, 4.0]
        beta_band = [17.0, 30.0]
        # with these thresholds for delta and beta the frequency decoder always enables stimulation
        # unknown specific range to start optimizing, better leave them as default and optimize the remaining ones
        delta_power_th = -10000000000000000000000000000000
        beta_power_th = 10000000000000000000000000000000
        # beta_change_th is not defined in the old one but is needed in the new one, so I found that the default value is 1
        beta_change_th = 1.0
    else:
        freq_power_buffer_size = int(parameters['DecoderMediator']['freq_power_buffer_size']['value'])
        delta_band = [float(parameters['DecoderMediator']['lower_delta_band']['value']),
                      float(parameters['DecoderMediator']['upper_delta_band']['value'])]
        beta_band = [float(parameters['DecoderMediator']['lower_beta_band']['value']),
                     float(parameters['DecoderMediator']['upper_beta_band']['value'])]
        delta_power_th = float(parameters['DecoderMediator']['delta_power_th']['value'])
        beta_power_th = float(parameters['DecoderMediator']['beta_power_th']['value'])
        beta_change_th = float(parameters['DecoderMediator']['beta_power_change_th']['value'])

    # PLL
    if is_old_data:
        pll_on_time = 6000
        waiting_time = 6000
    else:
        pll_on_time = int(parameters['DecoderMediator']['pll_on_time_ms']['value'])
        waiting_time = int(parameters['DecoderMediator']['waiting_time_ms']['value'])

    # Artifact (minimum) and down up thresholds for slow wave detection
    if is_old_data:
        down_up_th = -20.0
        threshold_min = -200-0
    else:
        down_up_th = float(parameters['DecoderMediator']['down_up_th']['value'])
        threshold_min = int(parameters['DecoderMediator']['threshold_min']['value'])

    # Target phase to stimulate
    target_phase = 10

    # ---------------------------------- Initialize decoders and start iterating ----------------------------------

    print("PROCESSING FILE...")
    if start_sample is None and end_sample is None:
        stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, time_stim_on_total, \
        n_detections, n_stimulations = decoders_process(info_ssd, channel, fs, signal, seq_eeg, ts_eeg, volume,
                                                        freq_power_buffer_size, lower_cutoff_freq,
                                                        upper_cutoff_freq, filter_order, delta_band, beta_band,
                                                        pll_on_time, waiting_time, delta_power_th, beta_power_th,
                                                        beta_change_th, threshold_min, threshold_max, down_up_th, k_pll,
                                                        f_nco, target_phase)
    else:
        signal = signal[:, start_sample:end_sample]
        info_ssd = info_ssd[:, start_sample:end_sample]
        
        #stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, time_stim_on_total, \
        #n_detections, n_stimulations = decoders_process_nrem_window(channel, fs, signal, seq_eeg, ts_eeg, volume,
        #                                                            lower_cutoff_freq, upper_cutoff_freq, filter_order,
        #                                                            pll_on_time, waiting_time, threshold_min,
        #                                                            threshold_max, down_up_th,
        #                             k_pll, f_nco, target_phase)
        stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, time_stim_on_total, \
        n_detections, n_stimulations = decoders_process(info_ssd, channel, fs, signal, seq_eeg, ts_eeg, volume,
                                                        freq_power_buffer_size, lower_cutoff_freq,
                                                        upper_cutoff_freq, filter_order, delta_band, beta_band,
                                                        pll_on_time, waiting_time, delta_power_th, beta_power_th,
                                                        beta_change_th, threshold_min, threshold_max, down_up_th, k_pll,
                                                        f_nco, target_phase)

    # --------------------------------------- Evaluation & results ---------------------------------------

    print("DONE")

    filtered_signal = filtering(signal[channel - 1, :], fs, lower_cutoff_freq, upper_cutoff_freq, filter_order)
    print("EVALUATING...")

    # Obtain metrics

    cm, csd, cmae, pas_up_phase, pas_not_up_phase = visualize_results(filtered_signal, target_phase,
                                                                      stimulus_positions, stim_on_off, fs,
                                                                      time_stim_on_total,
                                                                      down_up_before_stim_positions,
                                                                      minimum_before_stim_positions)
    metrics = [cm, csd, cmae, pas_up_phase, pas_not_up_phase, n_detections, n_stimulations]

    return metrics


def main():
    # metrics = process(dir="D:/BITBRAIN/Data for UZ/Recordings", file="_AR1_0", threshold_max=-60, k_pll=0.1, f_nco=1,
                      # start_sample=955167, end_sample=1460208)

    metrics = process(dir="D:/BITBRAIN/Data for UZ/Recordings", file="_AR1_0", threshold_max=-60, k_pll=0.1, f_nco=1)


if __name__ == "__main__":
    main()