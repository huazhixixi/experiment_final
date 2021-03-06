import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt


class Signal:

    def __init__(self, baudrate, samples, tx_symbols, fs, wavelength):
        '''
        :param baudrate: hz
        :param samples
        :param tx_symbols
        :param fs:hz
        :param wavelength: m

        '''
        self.baudrate = baudrate
        self.samples = samples
        self.tx_symbols = tx_symbols
        self.fs = fs
        self.wavelength = wavelength
        self.__samples_out_fiber = self.samples.copy()

    @property
    def original_samples(self):
        return self.__samples_out_fiber

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    @property
    def shape(self):
        return self.samples.shape

    @property
    def constl(self):
        return np.unique(self.tx_symbols[0])

    @property
    def symbol(self):
        return self.tx_symbols

    @property
    def sps(self):
        return int(self.fs / self.baudrate)

    def __len__(self):
        samples = np.atleast_2d(self.samples)
        return len(samples[0])

    def scatterplot(self, sps):
        import visdom
        fignumber = self.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=fignumber)
        for ith, ax in enumerate(axes):
            ax.scatter(self[ith, ::sps].real, self[ith, ::sps].imag, s=1, c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim(
                [self[ith, ::sps].real.min() - np.abs(self[ith, ::sps].real.min() / 2),
                 self[ith, ::sps].real.max() + np.abs(self[ith, ::sps].real.max() / 2)])
            ax.set_ylim(
                 [self[ith, ::sps].imag.min() - np.abs(self[ith, ::sps].imag.min() / 2),
                  self[ith, ::sps].imag.max() + np.abs(self[ith, ::sps].imag.max() / 2)])

        plt.tight_layout()
        plt.show()
    def inplace_normalise(self):

        self[:] = self[:]/np.sqrt(np.mean(np.abs(self[:])**2,axis=-1,keepdims=True))
        return self
    def resample(self,new_sps):
        import resampy
        self.samples = resampy.resample(self[:], self.fs / self.baudrate, new_sps)
        self.fs = new_sps * self.baudrate
        return self

class Fiber(object):

    def __init__(self, alpha, D, length, reference_wavelength, slope):
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
