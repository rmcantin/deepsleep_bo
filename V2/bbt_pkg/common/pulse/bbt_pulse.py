import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Pulse:
    position: np.ndarray
    code_raw: np.ndarray
    code_cor: np.ndarray


def bbt_pulse_decode(dig, sampling_rate, version=1, debug=False):
    """
Decodes a digital signal containing TTL pulses (version 1.00 by default)

Bitbrain has hardware components to allow two devices to be synchronized at sample level.
The concept is that one device (or an external component) sends a TTL pulse to the other device/s.
Specifically, we support the following two cases:
- Versatile Bio has a “Digital output” signal that, when enabled, sends TTL pulses than are received
  in the “Digital input” of another device. This case requires a component (TTL Trigger Bio cable)
  to connect the digital ports.
- An external component (TTL hyper scanning) sends TTL pulses that are received in the “Digital input” of two devices.

Version 1: Hamming TTL pulses (8-bits number every 4 seconds)
- One pulse every 4 seconds
- Each pulse is composed of two sequential bytes
- Each byte has the following structure:
  - Bit 1 set to 1 (used to detect the onset)
  - Bits 2 to 8 encode a Hamming(7,4) code, i.e., a 4-bits number (left-msb) to detect corrupt transmissions.
- The two 4-bits numbers are interpreted as an 8-bits number.

Version 0: Hamming TTL pulse every 8 seconds: only one byte (7-bits number, left-msb) with no error checking.

:param dig: digital signal
    - in most Bitbrain devices it is a 1-channel signal
    - in the Versatile 32 it is a 3-channel signal, with the TTL pulse in second channel
:param sampling_rate: sampling rate
:param version: version of the TTL pulse
    - version 1: 8-bytes codes every 4 seconds (2 bytes hamming codes)
    - version 0: 7-bytes codes every 8 seconds (1 byte, no error checking)
:param debug: enables debug mode

:returns dataclass with fields:
    - position: numpy array of positions (samples) where pulses are detected
    - code_raw: numpy array of codes decoded
    - code_cor: numpy array of codes corrected

Note that corrected codes are computed after the raw codes and
they reconstruct the codes by replacing the incorrect ones.
    """
    np.seterr(divide='ignore', invalid='ignore')

    pulse = Pulse(np.array([]), np.array([]), np.array([]))

    if dig.shape[0] > 1:
        dig = dig[1, :]   # case E32
    dig = np.squeeze(dig)

    # find onsets and remove those with not enough distance
    dig_onsets = np.squeeze(np.argwhere(dig > 0))
    if dig_onsets.size == 0:
        return pulse

    to_remove = np.argwhere(np.diff(dig_onsets) < 16) + 1
    dig_onsets = np.delete(dig_onsets, to_remove)

    # check last onset
    if (dig_onsets[-1] + 8) > dig.size:
        dig_onsets = np.delete(dig_onsets, -1)

    # iterate to decode hamming codes
    pulse.position = np.array(dig_onsets.size * [np.nan])
    pulse.code_raw = np.array(dig_onsets.size * [np.nan])
    pulse.code_cor = np.array(dig_onsets.size * [np.nan])
    for i in range(dig_onsets.size):
        if version == 1:
            # version 1: uses two hamming digits
            two_bytes = all([dig[dig_onsets[i]], dig[dig_onsets[i] + 8]])
            if two_bytes:
                byte_1 = dig[(dig_onsets[i]):(dig_onsets[i] + 8)]
                byte_2 = dig[(dig_onsets[i] + 8):(dig_onsets[i] + 16)]
                byte_1 = byte_1.reshape((-1, 1))    # transpose 1D array
                byte_2 = byte_2.reshape((-1, 1))    # transpose 1D array
                byte_1_err = ham_check(byte_1)
                byte_2_err = ham_check(byte_2)
                if not any([byte_1_err, byte_2_err]):
                    num_1 = ham_decode(byte_1)
                    num_2 = ham_decode(byte_2)
                    pulse.code_raw[i] = 16 * num_1 + num_2
            pulse.position[i] = dig_onsets[i]
        elif version == 0:
            # version 0: uses one digit
            byte = dig[(dig_onsets[i] + 1):(dig_onsets[i] + 8)]
            pulse.code_raw[i] = bi2de(byte)
            pulse.position[i] = dig_onsets[i]

    # apply robust statistics to compute the initial code
    if pulse.code_raw.size != 0:
        if version == 1:
            num_bytes = 8   # 4 per hamming byte
            seconds_period = 4
        elif version == 0:
            num_bytes = 7
            seconds_period = 8

        position_relative = pulse.position - pulse.position[0]
        num_pulse = np.round(position_relative / (sampling_rate * seconds_period))

        code_ini_dist = np.mod(pulse.code_raw - num_pulse, 2**num_bytes)
        values, counts = np.unique(code_ini_dist, return_counts=True)
        code_ini = values[np.argmax(counts)]
        pulse.code_cor = np.mod(np.arange(pulse.code_raw.size) + code_ini, 2**num_bytes)

    if debug:
        n_error = 100 * np.sum(pulse.code_raw != pulse.code_cor) / pulse.code_raw.size
        print('pulse error (%): {}'.format(n_error))

        p1, = plt.plot(pulse.position, pulse.code_cor)
        p2, = plt.plot(pulse.position, pulse.code_raw)
        plt.legend([p1, p2], ['corrected', 'raw'])
        plt.show()

    return pulse


def bi2de(binary):
    # MSB to the left
    decimal = 0
    for i in range(binary.size):
        if binary[i] == 1:
            decimal += 2**(binary.size - i - 1)
    return decimal


def ham_check(X):
    Hr = np.matrix('0 1 0 1 0 1 0 1; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1')
    error = bi2de(np.flipud(np.mod((Hr * X), 2)))
    return error


def ham_decode(X):
    Pr = np.matrix('0 0 0 1 0 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1')
    num = bi2de(np.flipud(Pr * X))
    return num
