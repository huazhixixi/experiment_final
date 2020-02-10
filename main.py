import resampy

from class_define import Signal
from dsp.dsp import remove_dc
from dsp.dsp import orthonormalize_signal
from dsp.dsp import cd_compensation
from dsp.dsp import FrequencyOffsetComp
from class_define import Fiber
from dsp.dsp import syncsignal
from tools.tools import find_each_section


class CoherentReceiver(object):

    def __init__(self,baudrate,adc_rate,samples,tx_symbols,tx_length,**param_dict):
        '''
        :param baudrate: The baudrate of the received signal [hz]
        :param adc_rate: The sampling rate of the scope in [hz]
        :param samples:  The received samples of the signal
        :param tx_length: The propagation length of the signal in KM
        :param param_dict:
            :key: ntaps: The ntaps for the LMS_equalizater
        '''
        signal = Signal(baudrate,samples,tx_symbols,adc_rate,wavelength=1550e-9)
        self.signal = signal
        self.span = Fiber(alpha=0.2,D=16.7,length=tx_length,reference_wavelength=1550,slope=0)
        self.param = param_dict

        self.dsp_processed_signals = []

    def front_end(self):
        self.signal = remove_dc(self.signal)
        self.signal.inplace_normalise()
        self.signal.resample(new_sps=2)
        self.signal = orthonormalize_signal(self.signal, os=1)

    def receiver_dsp(self):
        self.signal = cd_compensation(self.span, self.signal, self.signal.fs)
        freq_comp = FrequencyOffsetComp(8, True)
        self.signal = freq_comp.prop(self.signal)
        signals = find_each_section(self.signal, total_length=int(np.floor(len(self.signal) / self.signal.symbol.shape[1] / 2))-1, symbol_length= self.signal.symbol.shape[1])
        for signal in signals:
            signal = self.__equalization(signal)
            self.dsp_processed_signals.append(signal)

    def process(self):

        self.receiver_dsp()

    def __equalization(self,signal):
        from dsp.dsp import LMS, Superscalar
        from dsp.dsp import syncsignal_tx2rx as syncsignal

        lms = LMS(ntaps=self.param['ntaps'],lr=self.param['lr'],train_time=self.param['train_time'],train_symbols=signal.tx_symbols,loops=self.param['loops'])
        signal.inplace_normalise()
        signal = lms.equalize(signal)
        out = syncsignal(signal.samples, signal.tx_symbols)

        signal.tx_symbols = out
        signal.tx_symbols = signal.tx_symbols[:, :signal.shape[1]]
        cpe = Superscalar(256, 0.02, 20, 0, 4)

        signal = cpe.prop(signal)
        signal.inplace_normalise()
        return signal

def batch_equlization(signal):
    from dsp.dsp import LMS, Superscalar

    from dsp.dsp import syncsignal_tx2rx as syncsignal

   # cma = CMA(ntaps=321, lr=0.0001, loops=3)
    lms = LMS(ntaps=109,lr=[0.01/5],train_time=3,train_symbols=signal.tx_symbols,loops=3)
    signal[:] = signal[:] / np.sqrt(np.mean(np.abs(signal[:]) ** 2, axis=-1, keepdims=True))
    # signal = cma.equalize(signal)
    signal = lms.equalize(signal)
    # cma.plot_error()

    out = syncsignal(signal.samples, signal.tx_symbols)
    signal.tx_symbols = out
    signal.tx_symbols = signal.tx_symbols[:, :signal.shape[1]]
    cpe = Superscalar(256, 0.02, 20, 0, 4)
    signal = cpe.prop(signal)
    signal[:] = signal[:] / np.sqrt(np.mean(np.abs(signal[:]) ** 2, axis=-1, keepdims=True))
    noise = signal[:,1024:-1024] - signal.tx_symbols[:,1024:-1024]
    power = np.mean(np.abs(noise) ** 2, axis=-1).sum()
    print(10 * np.log10((2 - power) / power))
    return signal

def main(samples, symbols, adc_rate, dac_rate, baudrate, tx_symbol_length, span_length, span_param):
    span = Fiber(length=span_length, reference_wavelength=1550, slope=0, **span_param)

    sylen = tx_symbol_length * (baudrate / dac_rate)
    sylen = int(np.ceil(sylen))

    signal = Signal(baudrate, samples, symbols, adc_rate, wavelength=1550e-9)
    signal[:] = remove_dc(signal[:])
    signal[:] = signal[:] / np.sqrt(np.mean(np.abs(signal[:]) ** 2, axis=-1, keepdims=True))
    signal.samples = resampy.resample(signal.samples, signal.fs / signal.baudrate, 2)
    signal[:] = orthonormalize_signal(signal[:], os=1)

    signal.fs = 2 * signal.baudrate
    signal.tx_symbols = signal.tx_symbols[:, :sylen]
    # cdc
    signal = cd_compensation(span, signal, signal.fs)
    #
    freq_comp = FrequencyOffsetComp(8, True)
    signal = freq_comp.prop(signal)
    print(freq_comp.freq_offset)
    signals = find_each_section(signal, total_length=int(np.floor(len(signal) / sylen / 2))-1, symbol_length=sylen)
    for signal in signals:
        signal = batch_equlization(signal)
    return signals



def average_over_symbol(signals):
    xpol = 0
    ypol = 0

    for signal in signals:
        xpol = xpol + signal[0]
        ypol = ypol + signal[1]

    xpol = xpol/len(signals)
    ypol = ypol/len(signals)

    signal.samples = np.vstack((xpol,ypol))
    noise = signal[:, 1024:-1024] - signal.tx_symbols[:, 1024:-1024]
    power = np.mean(np.abs(noise) ** 2, axis=-1).sum()
    print(10 * np.log10((2 - power) / power))

if __name__ == '__main__':
    import numpy as np
    from tools.preprocessing import readdata
    from scipy.io import loadmat

    samples = readdata('xixi0')
    # samples = np.load('6.npz')['arr_0']
    # samples[0],samples[1] = samples[1],samples[0]
    symbols = loadmat('txqpsk.mat')['tx_symbols']
    # symbols[0] = np.conj(symbols[0])
    symbols[1] = np.conj(symbols[1])

    # symbols[0],symbols[1] = symbols[1],symbols[0]
    signals = main(samples, symbols, 100e9, 80e9, 20e9, span_length=320, tx_symbol_length=2 ** 18,
         span_param=dict(alpha=0.2, D=16.7))

    average_over_symbol(signals)
