import mne
import numpy as np
import matplotlib.pyplot as plt


def load_signals_simple(data: np.ndarray, num_channels: int, sampling_rate: int):
    """
    Imports one signal along with an event into MNE

    :param data: numpy array
    :param num_channels: number of channels
    :param sampling_rate: sampling rate
    :return mne object of raw data
    """
    channel_names = ["ch_{}".format(n + 1) for n in range(num_channels)]
    mne_raw = mne.io.RawArray(data, info=mne.create_info(channel_names, sampling_rate, 'misc'))
    return mne_raw


def load_signals(prop, signals, signal_id):
    """
    Imports one signal along with an event into MNE

    :param prop: properties of the signals and events (retrieved by Bitbrain functions)
    :param signals: signals (retrieved by Bitbrain functions)
    :param signal_id: signal id to importer (string)
    :return mne object of raw data
    """
    if signal_id not in prop.signals:
        print('signal not found: ' + signal_id)
        return

    num_channels = prop.signals[signal_id].num_channels
    sampling_rate = prop.signals[signal_id].sampling_rate
    return load_signals_simple(signals[signal_id].values, num_channels, sampling_rate)


def load(prop, log, signals, signal_id, events, event_id):
    """
    Imports one signal along with an event into MNE
    :param prop: properties of the signals and events (retrieved by Bitbrain functions)
    :param log: log of the signals and events (retrieved by Bitbrain functions)
    :param signals: signals (retrieved by Bitbrain functions)
    :param signal_id: signal id to importer (string)
    :param events: events (retrieved by Bitbrain functions)
    :param event_id: event id to importer (string)
    :return mne object of raw data and events

    Note that the timestamp of the events should be given in seconds and
    relative to the recording start.
    """
    if signal_id not in prop.signals:
        print('signal not found: ' + signal_id)
        return

    num_channels = prop.signals[signal_id].num_channels
    sampling_rate = prop.signals[signal_id].sampling_rate
    channel_names = ["ch_{}".format(n + 1) for n in range(num_channels)]
    mne_raw = mne.io.RawArray(signals[signal_id].values, info=mne.create_info(channel_names, sampling_rate, 'misc'))

    mne_events = np.empty([0, 3])
    if events is not None:
        if event_id not in prop.events:
            print('event not found ' + event_id)
        else:
            num_event_channels = prop.events[event_id].num_channels
            samples = np.rint(1e-6*(events[event_id].ts-log.t0)*sampling_rate)
            if samples.size > 0:
                for ch in range(num_event_channels):
                    ch_id = 'event_{}'.format(ch)
                    ch_values = np.rint(events[event_id].values[ch, :])

                    ch_events = np.vstack((samples, np.zeros(samples.shape), ch_values))
                    mne_events = np.vstack((mne_events, np.transpose(ch_events)))

        mne_raw.add_events(mne_events, channel_names[0])
    return mne_raw, mne_events


def view(mne_raw, mne_events=None, scale=120, filtering=None):
    """
    Visualizes a MNE object (use previous methods to load Bitbrain data into MNE objects)

    :param mne_raw: MNE object with raw data
    :param mne_events: MNE object with events
    :param scale: scale to visualize the signal (float)
    :param filtering: low-high corners of a bandpass filter to apply to the signal
    """
    mne_raw_filtered = mne_raw.copy()
    if filtering is not None:
        mne_raw_filtered.filter(filtering[0], filtering[1], picks='misc', method="iir",
                                iir_params=dict(order=4, ftype='butter', output="ba"))

    if mne_events is None:
        mne_raw_filtered.plot(scalings=scale, duration=7, remove_dc=False)
    else:
        mne_raw_filtered.plot(events=mne_events, scalings=scale, duration=7, remove_dc=False)
    plt.show(block=False)

    return plt
