import numpy as np
import os
import argparse
import math
import json
import matplotlib.pyplot as plt

from collections import Counter
from scipy.signal import butter, hilbert, lfilter
from auditory_stimulation_trigger import AuditoryStimulationEvaluation
from frequencies_detector import FrequenciesDetector
from slow_waves_decoder import SlowWavesDecoder

import bayesopt
from bayesoptmodule import BayesOptContinuous

n_detections = 0
n_stimulations = 0

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

def decoders_process(info_ssd, channel, fs, signal, volume, stim_probability, freq_power_buffer_size, lower_cutoff_freq,
                     upper_cutoff_freq, filter_order, delta_band, beta_band, pll_on_time, waiting_time, delta_power_th,
                     beta_power_th, threshold_min, threshold_max, down_up_th, k_pll, f_nco, target_phase):

    # Sleep stage decoder
    stim_on_ssd = info_ssd[0, :]

    # Initialize frequency detector
    fd = FrequenciesDetector(selected_channel=channel, sampling_rate=fs, delta_band=delta_band,
                             delta_th=delta_power_th, beta_band=beta_band, beta_th=beta_power_th,
                             buffer_size=freq_power_buffer_size, delta_decoder_on=True, beta_decoder_on=True)
    delta_power = np.repeat(-np.inf, signal.shape[1])
    stim_on_delta = np.zeros((signal.shape[1],))
    beta_power = np.repeat(-np.inf, signal.shape[1])
    stim_on_beta = np.ones((signal.shape[1],))

    # Initialize slow waves decoder
    sw_decoder = SlowWavesDecoder(selected_channel=channel, num_channels=signal.shape[0],
                                  sampling_rate=fs, amplitude_th=(threshold_min, threshold_max),
                                  volume=volume, pll_timings=(pll_on_time, waiting_time),
                                  filter_order=filter_order,
                                  lower_cutoff_freq=lower_cutoff_freq, upper_cutoff_freq=upper_cutoff_freq,
                                  stim_probability=stim_probability, k_pll=k_pll, f_nco=f_nco,
                                  target_phase=target_phase, down_up_th=down_up_th)
    stim_on_total = np.zeros((signal.shape[1],))
    stimulate = []
    detection_seq = []
    detection_samples = []
    down_up_before_stim = []
    minimum_before_stim = []

    # ...Iterate...
    step = 8
    epoch_length = step
    num_epochs = math.floor((signal.shape[1] - epoch_length) / step)
    seq_eeg = np.array(range(1, num_epochs + 2))
    ts_eeg = np.zeros((num_epochs + 1,))

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
            pll_output, detection_values, _, sw_updated = \
                sw_decoder.process(seq_eeg[epoch_i], signal[:, epoch_ini:epoch_end], ts_eeg[epoch_i])

            if sw_updated:
                stimulate.append(sw_decoder.detection[0, 0])
                detection_seq.append(sw_decoder.detection[5, 0])
                detection_samples.append(sw_decoder.detection[6, 0])
                down_up_before_stim.append(sw_decoder.detection[7, 0])
                minimum_before_stim.append(sw_decoder.detection[8, 0])

            # Stimulate ?
            if ssd_on and delta_on and beta_on:
                auto_mode_value = 1
                stim_on_total[epoch_ini:epoch_end] = 1
            else:
                auto_mode_value = 0
            sw_decoder.update_stimulation_on_off(auto_mode_value, auto_mode=True)

    ini_seq_num = np.nonzero(np.in1d(seq_eeg, np.asarray(detection_seq)))[0] * 8
    end_seq_num = ini_seq_num + 7

    stimulate_enabled_idx = np.where(np.asarray(stimulate) == 1)[0]
    stimulus_positions = (end_seq_num[stimulate_enabled_idx] - np.asarray(detection_samples)[stimulate_enabled_idx]) \
                         + 100 * 1e-3 * fs
    stimulus_positions = np.rint(stimulus_positions).astype(int)
    down_up_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(down_up_before_stim)[stimulate_enabled_idx]).astype(int)
    minimum_before_stim_positions = (end_seq_num[stimulate_enabled_idx] -
                                     np.asarray(minimum_before_stim)[stimulate_enabled_idx]).astype(int)

    print('Number of detections: ', len(detection_samples))
    print('Number of stimulations: ', len(stimulus_positions))

    global n_detections
    n_detections = len(detection_samples)
    global n_stimulations
    n_stimulations = len(stimulus_positions)

    # Events
    stim_on_total = np.zeros((1, stim_on_ssd.shape[0]))
    stim_on_total[0, np.where(np.count_nonzero(np.vstack([stim_on_ssd, stim_on_delta, stim_on_beta]) == 1,
                                               axis=0) == 3)[0]] = 1
    stim_on_off = np.vstack((stim_on_ssd, stim_on_delta, stim_on_beta, stim_on_total))

    time_stim_auto_gates_on = len(np.where(stim_on_total == 1)[0]) / fs

    return stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, \
           time_stim_auto_gates_on


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
    ase.plot_phase_distribution()
    ase.plot_mean_sem_segments(filtered_signal, (int(1.5 * fs), fs), point='stim')

    return ase.cm, ase.csd, ase.cmae, ase.pas_up_phase, ase.pas_not_up_phase


class BayesOptStimuli(BayesOptContinuous):

    def __init__(self, file_name, 
                 opt_mask = np.array([0,1,0,1,1,1]), 
                 alphas = np.array([0.1,0,1,1]),
                 lb = np.array([-250, -80, -80,  0.1, 0.8, 30]), 
                 dv = np.array([-200, -40, -20,  0.1, 1.5, 30]),
                 ub = np.array([-150, -20,   0,  1.0, 10,  90]),
                 dbd = False):

        # Problem dimension equals the number of parameters to optimize
        # This is deduced from the mask
        self.n = int(np.sum(opt_mask))

        # If mask is all ceros, optimization is not possible
        assert self.n > 0, "Mask is all zeros, no parameters will be optimized"

        super().__init__(self.n)

        # Prepare bounds for each parameter
        
        # Lower bounds for all parameters
        # Default values for all parameters
        # Upper bounds for all parameters

        print("Optimizing",self.n,"parameters")
        print("Mask:",opt_mask)
        print("Lower bounds:",lb)
        print("Default values:",dv)
        print("Upper bounds:",ub)

        self.lower_bound = lb[opt_mask>0]
        self.default_values = dv
        self.upper_bound = ub[opt_mask>0]
        self.opt_mask = opt_mask
        self.file_name = file_name
        self.alphas = alphas

        # Prepare arrays for metric recolection
        self.metrics = np.zeros(6)
        self.default_metrics = np.zeros(6)
        self.error_cases = []

        # Declare optimization mode:
        #
        # "dbd" -> "Divided By Default"- Each metric is divided by the value obtained 
        #                                from the default parameters.
        self.dbd = dbd

        # Data points for saving and plotting
        self.metric_dp = np.zeros((1,4))
        self.query_dp = np.zeros((1,6))

    def compute_default_metrics(self):

        self.evaluateSample(self.default_values[self.opt_mask>0])
        self.default_metrics = self.metrics

        return self.default_metrics

    def compute_alphas_dbd(self):
        
        self.alphas = np.array([self.alphas[0]/self.default_metrics[2],
                                self.alphas[1]/self.default_metrics[1],
                                self.alphas[2]/self.default_metrics[3],
                                self.alphas[3]/self.default_metrics[4]])

        return self.alphas

    def set_alphas(self, alphas):

        self.alphas = alphas

    def return_full_query(self,Xin):

        Q = self.default_values
        Q[self.opt_mask>0] = Xin

        return Q        

    def plot_metrics(self,fig_name):

        self.metric_dp = self.metric_dp[1:]

        init_it = self.parameters['n_init_samples']

        cmaes_init = self.metric_dp[:init_it,0]
        pups_init = self.metric_dp[:init_it,2]
        pnoups_init = self.metric_dp[:init_it,3]

        cmaes = self.metric_dp[init_it:,0]
        pups = self.metric_dp[init_it:,2]
        pnoups = self.metric_dp[init_it:,3]

        best_cmae = self.metric_dp[-1,0]
        best_pup = self.metric_dp[-1,2]
        best_pnotup = self.metric_dp[-1,3]

        fig, axs = plt.subplots(1,3,tight_layout=True)
        axs[0].plot(cmaes,-pups,'b.')
        axs[0].plot(cmaes_init,-pups_init,'b+')
        axs[0].plot(best_cmae,-best_pup,'r.')
        axs[0].set_xlabel('CMAE')
        axs[0].set_ylabel('-PUP')
        axs[1].plot(cmaes,pnoups,'b.')
        axs[1].plot(cmaes_init,pnoups_init,'b+')
        axs[1].plot(best_cmae,best_pnotup,'r.')
        axs[1].set_xlabel('CMAE')
        axs[1].set_ylabel('PnotUP')
        axs[2].plot(-pups,pnoups,'b.')
        axs[2].plot(-pups_init,pnoups_init,'b+')
        axs[2].plot(-best_pup,best_pnotup,'r.')
        axs[2].set_xlabel('-PUP')
        axs[2].set_ylabel('PnotUP')

        #plt.show()
        fig.set_size_inches(12.0,4.0)
        fig.savefig(fig_name,dpi=100)


    def evaluateSample(self, Xin):

        # ---------------------------------- Load & filtering ----------------------------------

        # Load npz file
        data = np.load(self.file_name, allow_pickle=True)
        print("LOADING FILE:",self.file_name)

        # Load signal, fs, and labels
        signal = data['x'].transpose(2, 0, 1).reshape(data['x'].shape[2], -1)
        fs = int(data['fs'][0])
        gt_labels = data['y']
        gt_labels_annot = resample_binarize_labels(gt_labels, fs)

        if '_ST2_' in self.file_name:  # No HB data after 497 min
            signal = signal[:, 0:497*60*fs]
            gt_labels_annot = gt_labels_annot[:, 0:497*60*fs]
        info_ssd = gt_labels_annot

        # Filter signal
        lower_cutoff_freq = 0.2
        upper_cutoff_freq = 5
        filter_order = 2

        # ---------------------------------- Define parameters ----------------------------------

        volume = 0.1
        stim_probability = 1
        channel = 10

        # Delta and beta
        freq_power_buffer_size = 4
        delta_band = [0.5, 4.0]
        beta_band = [17.0, 30.0]

        # PLL
        pll_on_time = 6000
        waiting_time = 6000

        # ---------------- Modifiable parameters
        # Delta and beta
        # with these thresholds for delta and beta the frequency decoder always enables stimulation
        # unknown specific range to start optimizing, better leave them as default and optimize the remaining ones
        delta_power_th = -10000000000000000000000000000000
        beta_power_th = 10000000000000000000000000000000

        # ---------------- Parameters to be optimized

        Q = self.default_values
        Q[self.opt_mask>0] = Xin

        print("QUERY:",Q)

        # # Minimum and maximum thresholds for slow wave detection
        # threshold_min = -200  # range (linear scale): [-250, -150]
        threshold_min = Q[0]
        # threshold_max = -40  # range (linear scale): [-80, -20]
        threshold_max = Q[1]
        # down_up_th = -20.0  # range (linear scale): [-80, 0]
        down_up_th = Q[2]

        # # PLL
        # k_pll = 0.1  # gain of the PLL, range (linear scale): [0.01, 0.5]
        k_pll = Q[3]
        # f_nco = 1.5  # number-controlled oscillator frequency, range (linear scale): [0.8, 10.0]
        f_nco = Q[4]
        # target_phase = 0  # target phase to stimulate, range (linear scale): [30, 90]
        target_phase = Q[5]

        # ---------------------------------- Initialize decoders and start iterating ----------------------------------

        print("PROCESSING FILE...")

        stimulus_positions, down_up_before_stim_positions, minimum_before_stim_positions, stim_on_off, time_stim_on_total\
        = decoders_process(info_ssd, channel, fs, signal, volume, stim_probability, freq_power_buffer_size,
                           lower_cutoff_freq, upper_cutoff_freq, filter_order, delta_band, beta_band, pll_on_time,
                           waiting_time, delta_power_th, beta_power_th, threshold_min, threshold_max, down_up_th, k_pll,
                           f_nco, target_phase)

        # --------------------------------------- Evaluation & results ---------------------------------------

        print("DONE")

        filtered_signal = filtering(signal[channel, :], fs, lower_cutoff_freq, upper_cutoff_freq, filter_order)

        print("EVALUATING...")

        # Obtain metrics
        try:

            cm, csd, cmae, pas_up_phase, pas_not_up_phase = visualize_results(filtered_signal, target_phase,
                                                                      stimulus_positions, stim_on_off, fs,
                                                                      time_stim_on_total, down_up_before_stim_positions,
                                                                      minimum_before_stim_positions)

            self.metrics = np.array([cm,csd,cmae,pas_up_phase,pas_not_up_phase,n_detections,n_stimulations])
            self.metric_dp = np.vstack((self.metric_dp,[cmae,csd,pas_up_phase,pas_not_up_phase]))
            self.query_dp = np.vstack((self.query_dp,Q))

            if np.isnan(self.metrics).any():
                raise TypeError("Encountered NaNs during evaluation")

            r = 2.0
            r = self.alphas[0]*cmae + self.alphas[1]*csd - self.alphas[2]*pas_up_phase + self.alphas[3]*pas_not_up_phase
            return r

        except:

            print("AN ERROR OCCURRED. RETURNING CONSTANT LIAR.")
            self.error_cases = self.error_cases + [Q.tolist()]
            self.metrics = np.zeros(6)
            
            r = 2.0
            return r

def main(args):

    # ---------------------------------- Obtain I/O arguments ------------------------------------

    data_dir = os.path.abspath(args.directory)
    file_id = "_"+args.file+"_"
    for file in os.listdir(data_dir):
        if file_id in file:
            file_name = file
            break
    file_path = os.path.join(data_dir,file_name)
    subject = file_name.split("_proc")[0]
        
    model_dir = os.path.abspath("../Models")
    model_name = "bopt_"+subject
    results_name = "results_"+subject

    opt_mode = "a"
    alphas = np.zeros(4)

    dbd = args.divbydef
    p = args.plot

    alpha_str = "_a"
    if len(args.alphas)!=4:
        print("ERROR. Please input 4 values for alphas.")
        return

    for i,a in enumerate(args.alphas):
        alphas[i] = float(a)
        alpha_str = alpha_str + "_" + a.replace(".","")
    model_name = model_name + alpha_str
    results_name = results_name + alpha_str

    if dbd:
        model_name = model_name + "_dbd"
        results_name = results_name + "_dbd"
    
    model_path = os.path.join(model_dir,model_name)
    print("MODEL WILL BE SAVED AS:",model_path)
    results_path = os.path.join(model_dir,results_name)
    print("RESULTS WILL BE SAVED AS:",results_path)

    # ---------------------------------- Problem definition ----------------------------------

    lb = np.zeros(6)
    dv = np.zeros(6)
    ub = np.zeros(6)
    opt_mask = np.zeros(6)

    if (len(args.lowbounds) != 6) or (len(args.defval) != 6) or (len(args.upbounds) != 6) or (len(args.optmask) !=6 ):
        print("Error. Please enter 6 values for bounds, mask and default values.")
        return

    for i in range(6):
        lb[i] = float(args.lowbounds[i])
        dv[i] = float(args.defval[i])
        ub[i] = float(args.upbounds[i])
        opt_mask[i] = int(args.optmask[i])

    n = np.sum(opt_mask)

    results = {}

    score = -1
    query = np.zeros(6)

    params = {}
    params['n_init_samples'] = int(args.init)
    params['n_iterations'] = int(args.iter)
    params['n_iter_relearn'] = 1
    params['l_type'] = 'mcmc'
    params['init_method'] = 2 # Sobol
    params['load_save_flag'] = 2 # Save, don't load
    params['save_filename'] = model_path + '.txt'
    params['verbose_level'] = 5 # Log
    params['log_filename'] = model_path + "_log.txt"

    bo = BayesOptStimuli(file_path,opt_mask,alphas,lb,dv,ub,dbd)
    bo.parameters = params
    default_metrics = bo.compute_default_metrics()
    if dbd:
        alphas = bo.compute_alphas_dbd()

    score, x_out, error = bo.optimize()
    query = bo.return_full_query(x_out)
    
    metrics = np.zeros(6)
    error_cases = bo.error_cases

    if not error:

        _ = bo.evaluateSample(x_out)
        best_metrics = bo.metrics
        metrics = bo.metric_dp[1:]
        queries = bo.query_dp[1:]

    np.savez(results_path+".npz",score=score,query=query,best_metrics=best_metrics,default_metrics=default_metrics,metrics_dp=metrics,queries_dp=queries)
    results= {"Query":query.tolist(),"Best Metrics":best_metrics.tolist(),"Default Metrics":default_metrics.tolist(),"Metric Datapoints":metrics.tolist(),"Query Datapoints":queries.tolist(),"Errors":error_cases}

    with open(results_path+".json",'w') as fp:
        json.dump(results,fp)

    if(p):
        fig_name = results_path + '.png'
        print("SAVING FIGURE AT:",fig_name)
        bo.plot_metrics(fig_name)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-f","--file",required=True,help="Identificador del fichero a optimizar. (e.g. 'ST3')")
    ap.add_argument("-d","--directory",default="../Data",help="Directorio con el fichero a optimizar.")
    ap.add_argument("-a","--alphas",default=['0.1','0.0','1.0','1.0'],nargs='+',help="Coeficientes para la suma pesada. Se requieren 4: CMAE, CSD, PUP y PnotUP")
    ap.add_argument("-lb","--lowbounds",default=['-250','-80','-80','0.1','0.8','30'],nargs='+',help="Límites inferiores del espacio de exploración. Se requieren 6: th_min, th_max, down_up_th, k_pll, f_nco y target_phase.")
    ap.add_argument("-dv","--defval",default=['-200','-40','-20','0.1','1.5','30'],nargs='+',help="Valores por defecto de los parámetros. Se requieren 6, igual que lb.")
    ap.add_argument("-ub","--upbounds",default=['-150','-20','0','1.0','10','90'],nargs='+',help="Límites superiores del espacio de exploración. Se requieren 6, igual que lb.")
    ap.add_argument("-op","--optmask",default=['0','1','0','1','1','1'],nargs='+',help="Máscara de optimización. Se requieren 6, igual que lb. Los parámetros con 0 usarán siempre dv, los parámetros con más de 0 serán optimizados.")
    ap.add_argument("-in","--init",default=16,help="Iteraciones para construcción del modelo inicial (exploración, no optimización)")
    ap.add_argument("-it","--iter",default=32,help="Iteraciones del proceso de optimización.")
    ap.add_argument("-dbd","--divbydef",action="store_true",help="Dividir métricas por valor por defecto. (Se recomienda usarlo)")
    ap.add_argument("-p","--plot",action="store_true",help="Dibujar métricas en los puntos explorados al final.")

    args = ap.parse_args()
        
    main(args)