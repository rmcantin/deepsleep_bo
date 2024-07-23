import numpy as np
import matplotlib.pyplot as plt
from bbt_pkg.common.pulse import bbt_pulse


def hw_sync(log, prop, signals, is_platform, pulse_version, debug):

    padding = False
    # set to true to add padding to the max data position
    # set to false to remove up to the min data position

    # constants
    device_type_bio = 'BIO'
    signal_type_din = 'DIN'
    signal_type_dout = 'DOUT'
    if is_platform:
        device_type_bio = 'bth_biosensing'
        signal_type_din = 'din'
        signal_type_dout = 'dout'

    # we remove the initial padding added by bbt export process
    for signal_id in prop.signals:
        pad_ini = log.signals[signal_id].pad_ini
        signals[signal_id].values = np.delete(signals[signal_id].values, slice(0, pad_ini), axis=1)
        log.signals[signal_id].t0 += 1e6 * (pad_ini / log.signals[signal_id].sr_no)

    # separate for each device
    map_device_signals = dict()
    for signal_id in prop.signals:
        device_name = prop.signals[signal_id].device_name
        if device_name in map_device_signals:
            map_device_signals[device_name].append(signal_id)
        else:
            map_device_signals[device_name] = [signal_id]

    device_list = list(map_device_signals.keys())
    if len(device_list) <= 1:
        print('hw_sync expects one Biosensing device and an EEG device with Din')
        return

    # parse the din-dout pulses
    dig_pulse = []
    for device_idx in range(0, len(device_list)):
        device_name = device_list[device_idx]
        device_type = prop.signals[map_device_signals[device_name][0]].device_type
        if device_type == device_type_bio:
            prop_list = [prop.signals[signal_id] for signal_id in map_device_signals[device_name]]
            prop_dig = list(filter(lambda item: item.signal_type == signal_type_dout, prop_list))
            if len(prop_dig) == 1:
                prop_dig = next(iter(prop_dig))
                dout_sr = prop_dig.sampling_rate
                dig_pulse.append(bbt_pulse.bbt_pulse_decode(signals[prop_dig.id].values, prop_dig.sampling_rate, pulse_version, debug))
        else:
            prop_list = [prop.signals[signal_id] for signal_id in map_device_signals[device_name]]
            prop_dig = list(filter(lambda item: item.signal_type == signal_type_din, prop_list))
            if len(prop_dig) == 1:
                prop_dig = next(iter(prop_dig))
                dig_pulse.append(bbt_pulse.bbt_pulse_decode(signals[prop_dig.id].values, prop_dig.sampling_rate, pulse_version, debug))

    empty_pulses = list(map(lambda item: len(item.code_raw) == 0, dig_pulse))
    if not any(empty_pulses):
        print('- b_hw_sync: sync signals')

        # find the maximum initial code received
        max_initial_code = max(list(map(lambda item: item.code_cor[0], dig_pulse)))

        # compute the sample position where that code was observed
        init_code_pos = np.empty((len(dig_pulse)))
        for device_idx in range(0, len(dig_pulse)):
            idx_code = np.where(max_initial_code == dig_pulse[device_idx].code_cor)[0]
            init_code_pos[device_idx] = dig_pulse[device_idx].position[idx_code]

        # compute the offset to be added
        # we correct the t0 estimate with the removed samples
        for device_idx in range(0, len(dig_pulse)):
            device_name = device_list[device_idx]
            signal_list = map_device_signals[device_name]
            if padding:
                max_offset = max(init_code_pos)
                t_to_add = (max_offset - init_code_pos[device_idx]) / dout_sr
                print('-- t_to_add to device {} (ms): {}'.format(device_name, t_to_add*1e3))
                for signal_id in signal_list:
                    samples_to_add = round(t_to_add * prop.signals[signal_id].sampling_rate)
                    signals[signal_id].values = np.concatenate((np.full([signals[signal_id].values.shape[0], samples_to_add], np.nan), signals[signal_id].values), axis=1)
                    pad_ini = samples_to_add
                    log.signals[signal_id].t0 -= 1e6 * (pad_ini / log.signals[signal_id].sr_no)
            else:
                min_offset = min(init_code_pos)
                t_to_rem = (init_code_pos[device_idx] - min_offset) / dout_sr
                print('-- t_to_rem to device {} (ms): {}'.format(device_name, t_to_rem * 1e3))
                for signal_id in signal_list:
                    samples_to_rem = round(t_to_rem * prop.signals[signal_id].sampling_rate)
                    if samples_to_rem > 0:
                        signals[signal_id].values = np.delete(signals[signal_id].values, slice(0, samples_to_rem), axis=1)
                    pad_ini = -samples_to_rem
                    log.signals[signal_id].t0 -= 1e6 * (pad_ini / log.signals[signal_id].sr_no)

    if debug:
        for device_idx in range(0, len(device_list)):
            device_name = device_list[device_idx]
            device_type = prop.signals[map_device_signals[device_name][0]].device_type
            if device_type == device_type_bio:
                prop_list = [prop.signals[signal_id] for signal_id in map_device_signals[device_name]]
                prop_dig = list(filter(lambda item: item.signal_type == signal_type_dout, prop_list))
                if len(prop_dig) == 1:
                    prop_dig = next(iter(prop_dig))
                    dout_sr = prop_dig.sampling_rate
                    dout_sig = signals[prop_dig.id].values
                    dout_t = np.arange(dout_sig.shape[1]) / dout_sr
                    p1, = plt.plot(dout_t, 0.9*np.squeeze(dout_sig))
            else:
                prop_list = [prop.signals[signal_id] for signal_id in map_device_signals[device_name]]
                prop_dig = list(filter(lambda item: item.signal_type == signal_type_din, prop_list))
                if len(prop_dig) == 1:
                    prop_dig = next(iter(prop_dig))
                    din_sr = prop_dig.sampling_rate
                    din_sig = signals[prop_dig.id].values
                    din_t = np.arange(din_sig.shape[1]) / din_sr
                    if din_sig.shape[0] > 1:
                        din_sig = din_sig[1, :]  # case E32
                    p2, = plt.plot(din_t, np.squeeze(din_sig))
        plt.legend([p1, p2], ['dout', 'din'])
        plt.show()

    return signals, log
