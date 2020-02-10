import numpy as np
from class_define import Signal
from dsp.dsp import syncsignal


def average_over_symbol(signals:[Signal]) -> Signal:
    if len(signals) ==0:
        return
    xpol = 0
    ypol = 0

    for signal in signals:
        xpol = xpol + signal[0]
        ypol = ypol + signal[1]

    xpol = xpol/len(signals)
    ypol = ypol/len(signals)


    samples = np.vstack((xpol,ypol))
    signal = Signal(signals[0].baudrate,samples,signals[0].tx_symbols,signals[0].fs,signals[0].wavelength)
    return signal


def find_each_section(signal, total_length, symbol_length,is_visable):
    # 2e6 ----5 被采样，降到2倍采样
    out, corr_res = syncsignal(signal.tx_symbols, signal.samples, 2, is_visable)
    signals = []

    for i in range(total_length):
        signals.append(Signal(signal.baudrate, out[:, :symbol_length * signal.sps],

                              signal.tx_symbols, signal.fs, signal.wavelength))

        try:
            out = out[:, symbol_length * signal.sps:]
            out, _ = syncsignal(signal.symbol, out, signal.sps, is_visable)
        except Exception as e:
            return signals

    return signals

import matplotlib.pyplot as plt
def scatterplot(samples,sps):
        import visdom
        fignumber = samples.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=fignumber)
        for ith, ax in enumerate(axes):
            ax.scatter(samples[ith, ::sps].real, samples[ith, ::sps].imag, s=1, c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim(
                [samples[ith, ::sps].real.min() - np.abs(samples[ith, ::sps].real.min() / 2),
                 samples[ith, ::sps].real.max() + np.abs(samples[ith, ::sps].real.max() / 2)])
            ax.set_ylim(
                 [samples[ith, ::sps].imag.min() - np.abs(samples[ith, ::sps].imag.min() / 2),
                  samples[ith, ::sps].imag.max() + np.abs(samples[ith, ::sps].imag.max() / 2)])

        plt.tight_layout()
        plt.show()
