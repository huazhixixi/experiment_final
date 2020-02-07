import numpy as np
class Signal:

    def __init__(self,baudrate,samples,tx_symbols,fs,wavelength):
        self.baudrate = baudrate
        self.samples = samples
        self.tx_symbols = tx_symbols
        self.fs = fs
        self.wavelength = wavelength

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    @property
    def shape(self):
        return self.samples.shape


    def __len__(self):

        samples = np.atleast_2d(self.samples)
        return len(samples[0])

class Fiber(object):

    def __init__(self, alpha, D, length, reference_wavelength,slope):
        '''
            :param alpha:db/km
            :D:s^2/km
            :length:km
            :reference_wavelength:nm
        '''
        self.alpha = alpha
        self.D = D
        self.length = length
        self.reference_wavelength = reference_wavelength  # nm
        self.slope = slope

    @property
    def alphalin(self):
        alphalin = self.alpha / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''
        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wavelength * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength * 1e-12 * self.D + (
                self.reference_wavelength * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        '''
        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length



