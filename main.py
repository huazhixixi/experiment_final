import numpy as np
import resampy

from class_define import Signal
from dsp.dsp import remove_dc
from dsp.dsp import orthonormalize_signal
from dsp.dsp import cd_compensation
from dsp.dsp import FrequencyOffsetComp
from class_define import Fiber
from dsp.dsp import syncsignal

def find_each_section(signal,total_length,symbol_length):
    #2e6 ----5 被采样，降到2倍采样
    import matplotlib.pyplot as plt
    from scipy.signal import correlate
    out,corr_res = syncsignal(signal.tx_symbols,signal.samples,2,1)
    signals = []
    for i in range(total_length):
        signals.append(Signal(signal.baudrate,out[:,i*symbol_length*signal.sps:(i+1)*symbol_length*signal.sps],

                              signal.tx_symbols,signal.fs,signal.wavelength))

        out = out[:,(i + 1) * symbol_length * signal.sps:]
        out ,_=  syncsignal(out,signal.samples,2,1)

    return signals


def batch_equlization(signal):
    from dsp.dsp import CMA,LMS,Superscalar
    from dsp.dsp import syncsignal_tx2rx as syncsignal

    cma = CMA(ntaps=321, lr=0.0001, loops=3)
    signal[:] = signal[:]/np.sqrt(np.mean(np.abs(signal[:])**2,axis=-1,keepdims=True))
    signal = cma.equalize(signal)
    # cma.plot_error()

    out = syncsignal(signal.samples,signal.tx_symbols)
    signal.tx_symbols = out
    signal.tx_symbols = signal.tx_symbols[:,:signal.shape[1]]
    cpe = Superscalar(256,0.02,20,0,4)
    signal = cpe.prop(signal)
    signal[:] = signal[:]/np.sqrt(np.mean(np.abs(signal[:])**2,axis=-1,keepdims=True))
    noise = signal[:] - signal.tx_symbols
    power = np.mean(np.abs(noise)**2,axis=-1).sum()
    print(10*np.log10((2-power)/power))
def main(samples,symbols,adc_rate,dac_rate,baudrate,tx_symbol_length,span_length,span_param):

    span = Fiber(length=span_length,reference_wavelength=1550,slope=0,**span_param)

    sylen = tx_symbol_length *(baudrate/dac_rate)
    sylen = int(np.ceil(sylen))

    signal = Signal(baudrate,samples,symbols,adc_rate,wavelength=1550e-9)
    signal[:] = remove_dc(signal[:])
    signal[:] = signal[:]/np.sqrt(np.mean(np.abs(signal[:])**2,axis=-1,keepdims=True))
    signal.samples = resampy.resample(signal.samples,signal.fs/signal.baudrate,2)
    signal[:] = orthonormalize_signal(signal[:],os=1)

    #signal[1] = signal[1].imag + 1j*signal[1].real
    # signal[1] = np.conj(signal[1])
    signal.fs =  2*signal.baudrate
    signal.tx_symbols = signal.tx_symbols[:,:sylen]
    # cdc
    signal = cd_compensation(span,signal,signal.fs)
    #
    freq_comp = FrequencyOffsetComp(8,True)
    signal = freq_comp.prop(signal)
    print(freq_comp.freq_offset)
    signals = find_each_section(signal,total_length=int(np.floor(len(signal)/sylen/2)),symbol_length=sylen)
    for signal in signals:
        batch_equlization(signal)
if __name__ == '__main__':
    import numpy as np
    import joblib
    from preprocessing import readdata
    from scipy.io import loadmat
    samples = readdata('xixi0')
    #samples = np.load('6.npz')['arr_0']
    #samples[0],samples[1] = samples[1],samples[0]
    symbols = loadmat('txqpsk.mat')['tx_symbols']
    #symbols[0] = np.conj(symbols[0])
    symbols[1] = np.conj(symbols[1])


    #symbols[0],symbols[1] = symbols[1],symbols[0]
    main(samples,symbols,100e9,80e9,20e9,span_length=320,tx_symbol_length=2**18,span_param=dict(alpha=0.2,D=16.7))


# 符号y 共轭，其余不变
# y共轭。 样点偏振呼唤
# y共轭。不变，x取共轭
