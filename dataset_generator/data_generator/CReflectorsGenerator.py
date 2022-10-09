import numpy as np
from enum import Enum
from scipy.signal import chirp
import colorednoise as cn


class noise_type(Enum):
    white = 0
    pink = 1
    brownian = 2


class CReflectorsGenerator:
    def __init__(self, max_targets: int):
        self.max_targets = max_targets
        self.prev_echoes: np.ndarray = np.zeros(self.max_targets)
        self.prev_echoes = np.vstack((self.prev_echoes, np.zeros(self.max_targets)))
        pass

    def clean(self):
        self.prev_echoes = np.zeros(self.max_targets)
        self.prev_echoes = np.vstack((self.prev_echoes, np.zeros(self.max_targets)))

    def create_reflectors_waveform(self, range_list_in_meters: np.ndarray, attenuation: np.ndarray, snr_ratio: float):
        data_valid = True
        equal_len = np.shape(range_list_in_meters) == np.shape(attenuation)
        assert equal_len, "size of inputs must be the same"

        list_size = np.size(range_list_in_meters)
        more_than_max_targets = list_size <= self.max_targets
        assert more_than_max_targets
        # trim decimal point assume 0.1 meter resolution
        # cause target collapse two targets may collapse to one and this will cause problem in training
        # based on sample frequency
        sample_freq_hz = 300000
        # sound speed in air m / s
        Vp = 343
        # radar range equation R = 0.5 * Vp * Delay
        delays_sec = range_list_in_meters * 2 / Vp

        # based on number of zeros in the sample freq
        max_decimals = 4  # 5
        range_list_in_meters_rounded = np.around(range_list_in_meters, decimals=max_decimals)
        range_list_in_meters_orig = range_list_in_meters
        range_list_in_meters = range_list_in_meters_rounded
        # patch for the above problem
        range_list_in_meters_unique = np.unique(range_list_in_meters)
        if not (len(range_list_in_meters_unique) == len(range_list_in_meters)):
            data_valid = False
        range_list_in_meters = range_list_in_meters_unique
        # sort ranges to match waveform
        range_list_in_meters = np.sort(range_list_in_meters)
        # reshape data in case duplicated values were removed
        attenuation = attenuation[:len(range_list_in_meters)]
        echo_locations = (delays_sec * sample_freq_hz).astype(int) - 1
        echo_locations_unique = np.unique(echo_locations)
        if not (len(echo_locations_unique) == len(echo_locations)):
            data_valid = False
        # append zeros in order to add to numpy array
        padding = self.max_targets - len(echo_locations)
        echo_locations_padded = np.pad(echo_locations, (0, padding), "constant")

        # check if we generated this range before , data should be unique for training
        if np.shape(self.prev_echoes)[0] > 0 and data_valid:
            is_in_list = np.any(np.all(np.equal(self.prev_echoes, echo_locations_padded), axis=1))
            if is_in_list:
                data_valid = False
            else:
                self.prev_echoes = np.vstack((self.prev_echoes, echo_locations_padded))
                pass

        # project assumption
        max_range_meters = 10
        max_delay_sec = 2 * max_range_meters / Vp
        # Tx signal properties

        signal_time_sec = 3e-3
        fmin_hz = 100000
        fmax_hz = 30000
        tx_time_sec_vec = np.arange(0, signal_time_sec, 1 / sample_freq_hz)
        # signal model is pulsed lfm down, single pulse assumption
        tx_signal = chirp(tx_time_sec_vec, fmin_hz, signal_time_sec, fmax_hz, method='linear')

        # reflection  model
        delay_vec_len = int(sample_freq_hz * max_delay_sec)  # max_range / sample_time
        Echo_deltas = np.zeros(delay_vec_len)

        if data_valid:
            Echo_deltas[echo_locations] = attenuation

        # conv with discrete delta functions
        echoes_no_noise = np.convolve(Echo_deltas, tx_signal)
        echoes = self.add_colored_noise(echoes_no_noise, snr_ratio, noise_type.white)
        # fill mask based on echo location , no overlap single label
        tx_of_ones = np.ones_like(tx_signal)
        # fill mask with overlap
        seg_mask = np.convolve(Echo_deltas, tx_of_ones)
        # append zeroes to range_list_in_meters if required , zero means no target
        zero_to_append = self.max_targets - np.size(range_list_in_meters)
        range_list_in_meters_fixed_size = np.pad(range_list_in_meters, (0, zero_to_append), 'constant')
        # convert true , false to 0 ,1
        seg_mask = np.multiply(seg_mask, 1)
        return tx_signal, echoes, range_list_in_meters_fixed_size, seg_mask, data_valid

    def add_colored_noise(self, signal, snr_ratio, type: noise_type):
        beta = type.value
        sig_avg_watts = np.mean(signal ** 2)
        required_noise_avg_watts = sig_avg_watts / snr_ratio
        # Generate noise
        noise = cn.powerlaw_psd_gaussian(beta, len(signal))
        noise_avg_watts = np.mean(noise ** 2)
        scale_factor = np.sqrt(required_noise_avg_watts / noise_avg_watts)
        noise = noise * scale_factor

        # Noise up the original signal
        noisy_signal = signal + noise
        return noisy_signal
