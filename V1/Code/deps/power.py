import numpy as np
from scipy.signal.windows import hamming, hann, boxcar
from scipy.signal import periodogram
from typing import List, Tuple


def welch_bbt(x: np.ndarray, window_type: str, window_size: int, noverlap: int, nfft: int, fs: int,
              channels: List[int], max_freq: int, average_freq_bins: bool, log10: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Power Spectral Density estimate via Welch's method.

    Same as Matlab function [Pxx,F] = pwelch(x,window,noverlap,nfft,fs) but without averaging all bins

    Parameters
    ----------
    x : numpy.ndarray
        signal data (channels x time)
    window_type : str
        accepted window types (hamming, hann, rect)
    window_size : int
        size of the window to compute each periodogram
    noverlap : int
        number of non-overlapping samples between windows
    nfft : int
        number of points to compute the PSD (fs / nfft determines the resolution)
    fs : int
        sampling rate
    channels : list of int
        index of the channels for which the PSD is calculated
    max_freq : int
        highest frequency to compute the PSD
    average_freq_bins : bool
        if True average frequency bins

    Returns
    -------
    pxx
        PSD result (freqs x time x channels) with time in window bins
        if average_freq_bins --> PSD result (freqs x channels)
    f
        frequency corresponding with each value of the first dimension of pxx

    """

    # Check input
    if x.ndim > 2:
        raise Exception('x must be a 2d numpy array')

    # window_type: hamming, hann or rect
    if window_type == 'hamming':
        w = hamming(window_size, sym=False)  # Periodic
    elif window_type == 'hann':
        w = hann(window_size, sym=False)
    elif window_type == 'rect':
        w = boxcar(window_size)
    else:
        raise Exception('window_type must be hamming, hann or rect')

    if x.ndim == 2:
        length_eeg = x.shape[1]
    elif x.ndim == 1:
        length_eeg = x.shape[0]
    else:
        raise Exception('only 1-d or 2-d signals allowed as inputs')

    resolution = fs / nfft
    num_bins = int((max_freq / resolution) + 1)
    overlap = window_size - noverlap
    f = np.arange(0, max_freq+resolution, resolution)

    # Checking data
    if (window_size % noverlap) != 0:
        raise Exception('Error in MyWelch: window_size not multiple of noverlap')

    if (nfft % window_size) != 0:
        raise Exception('Error in MyWelch: nfft not multiple of window_size')

    if (max_freq % resolution) != 0:
        raise Exception('Error in MyWelch: max_freq not multiple of resolution')

    # Init Pxx
    num_computations = (np.floor((length_eeg - window_size) / overlap) + 1).astype(int)
    pxx = np.zeros((num_bins, num_computations, len(channels)))

    # Periodogram a full...
    for c in range(0, len(channels)):
        channel = channels[c]
        for i in range(0, num_computations):
            idx1 = i * overlap
            idx2 = idx1 + window_size
            if x.ndim == 2:
                _, p = periodogram(x=x[channel, idx1:idx2], window=w, nfft=nfft, fs=fs, detrend=False)
            elif x.ndim == 1:
                _, p = periodogram(x=x[idx1:idx2], window=w, nfft=nfft, fs=fs, detrend=False)
            else:
                raise Exception('only 1-d or 2-d signals allowed as inputs')
            pxx[:, i, c] = p[0:num_bins]

    # Log form
    if log10:
        pxx = 10 * np.log10(pxx)

    # Average frequency bins
    if average_freq_bins:
        pxx = np.nanmean(pxx, axis=1)

    return pxx, f


def welch_sleep_stages(eeg: np.ndarray, fs, sleep_stages: List[int], log10: bool, eeg_ok: np.ndarray):

    # 0 'Wake'
    # 1 'N1'
    # 2 'N2'
    # 3 'N3'
    # 4 'REM'

    num_stages = 5
    psds_stage = np.zeros((num_stages, 121, eeg.shape[0]))
    psds_stage_no_noise = np.zeros((num_stages, 121, eeg.shape[0]))
    coumt_stages = [sleep_stages.count(x) for x in range(0, num_stages)]
    psd_stage = [np.zeros((coumt_stages[x], 121, eeg.shape[0])) for x in range(0, num_stages)]
    cont = np.zeros((5,)).astype(int)
    for lab_idx, lab in enumerate(sleep_stages):
        if lab < 5 and lab >= 0:
            psd_st, freqs_st = welch_bbt(x=eeg[:, int(lab_idx * 30 * fs):int((lab_idx + 1) * 30 * fs)],
                                         window_type='hamming', window_size=int(fs), noverlap=int(fs / 2),
                                         nfft=int(4 * fs), fs=int(fs),
                                         channels=list(range(0, eeg.shape[0])), max_freq=30, average_freq_bins=False,
                                         log10=False)
            if log10:
                psd_st = 10 * np.log10(psd_st)
            psd_stage[lab][cont[lab], :, :] = np.nanmean(psd_st, axis=1)
            cont[lab] += 1
    for lab in range(0, 5):
        psds_stage[lab, :, :] = np.nanmean(psd_stage[lab], axis=0)
        for ch in range(0, psd_stage[lab].shape[2]):
            ok_channel = eeg_ok[ch, [i for i in range(0, len(sleep_stages)) if sleep_stages[i] == lab]]
            psds_stage_no_noise[lab, :, ch] = np.nanmean(psd_stage[lab][ok_channel, :, ch], axis=0)

    return psds_stage, psds_stage_no_noise, freqs_st


class ERD:
    """ Event Related Desynchronization

    This class stores ERD related information and process ERD

    Attributes
    ----------
    trials : np.ndarray
        eeg segments in channels x time x num_trials form
    sampling_rate : int
        number of samples per second

    Methods
    -------
    compute_time_frequency(baseline_time: float) :meth:`bbt_ds.decoding.power.ERD.compute_time_frequency`
        computes ERD with respect to baseline
    plot_erd()
        plots the ERD
    compute_numerical_erd(self, freqs, times)

    """

    def __init__(self, trials: np.ndarray, sampling_rate: int):
        self.trials = trials
        self.sampling_rate = sampling_rate

    def compute_time_frequency(self, baseline_time: float):
        """ Computes ERD as frequency changes over time related to the average frequency during baseline time before
        onset.

        Parameters
        ----------
        baseline_time : float
            seconds before trial onset used to compute the baseline to reference the desynchronization

        """

        pass

    def plot_erd(self):
        """ Plot ERD

        """

        pass

    def compute_numerical_erd(self, freqs, times):
        """ Get numerical value for the ERD.

        Parameters
        ----------
        baseline_time : float
            seconds before trial onset used to compute the baseline to reference the desynchronization

        """

        pass
