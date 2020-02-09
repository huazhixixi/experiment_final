import numpy as np
from class_define import Signal
from dsp.dsp import syncsignal


def average_over_symbol(signals:[Signal]) -> Signal:
    xpol = 0
    ypol = 0

    for signal in signals:
        xpol = xpol + signal[0]
        ypol = ypol + signal[1]

    xpol = xpol/len(signals)
    ypol = ypol/len(signals)


    samples = np.vstack((xpol,ypol))
    signal = Signal(signals[0].baudrate,samples,signals[0].tx_symbols,signals[0].fs,signals[0].wavelenght)
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