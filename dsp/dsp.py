import numpy as np
from numpy.fft import fftshift

from class_define import Signal
from dsp.dsp_tools import _segment_axis
from dsp.numba_core import cma_equalize_core, lms_equalize_core
from dsp.filter_design import rrcos_pulseshaping_freq
import matplotlib.pyplot as plt

def cd_compensation(span, signal: Signal, fs) -> Signal:
    '''
        span: The span for cd_c,should inlcude the following attributes:
            beta2:callable: receive the signal wavelength and return beta2
        signal:
            in place modify the signal
    '''

    center_wavelength = signal.wavelength
    freq_vector = np.fft.fftfreq(len(signal[0]), 1 / fs)
    omeg_vector = 2 * np.pi * freq_vector
    if not isinstance(span, list):
        spans = [span]
    else:
        spans = span

    for span in spans:
        beta2 = -span.beta2(center_wavelength)
        dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span.length
        for row in signal[:]:
            row[:] = np.fft.ifft(np.fft.fft(row) * np.exp(dispersion))

    return signal


def matched_filter(signal: Signal, roll_off) -> Signal:
    samples = np.copy(signal[:])
    for row in samples:
        row[:] = rrcos_pulseshaping_freq(row, signal.fs, 1 / signal.baudrate, roll_off)

    return Signal(signal.baudrate, samples, signal.tx_symbols, signal.fs, signal.wavelength)


class Equalizer(object):
    def __init__(self, ntaps, lr, loops,backend='mpl'):
        self.backend = backend
        self.wxx = np.zeros((1, ntaps), dtype=np.complex)
        self.wxy = np.zeros((1, ntaps), dtype=np.complex)

        self.wyx = np.zeros((1, ntaps), dtype=np.complex)

        self.wyy = np.zeros((1, ntaps), dtype=np.complex)

        self.wxx[0, ntaps // 2] = 1
        self.wyy[0, ntaps // 2] = 1

        self.ntaps = ntaps
        self.lr = lr
        self.loops = loops
        self.error_xpol_array = None
        self.error_ypol_array = None

        self.equalized_symbols = None

    def equalize(self, signal):

        raise NotImplementedError

    def scatterplot(self, sps=1):
        import matplotlib.pyplot as plt
        fignumber = self.equalized_symbols.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=fignumber)
        
        for ith, ax in enumerate(axes):
            ax.scatter(self.equalized_symbols[ith, ::sps].real, self.equalized_symbols[ith, ::sps].imag, s=1, c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim([self.equalized_symbols[ith, ::sps].real.min() - self.equalized_symbols[ith, ::sps].real.min()/4 , 
                         self.equalized_symbols[ith, ::sps].real.max() + self.equalized_symbols[ith, ::sps].real.max()/4])
            ax.set_ylim([self.equalized_symbols[ith, ::sps].imag.min() - self.equalized_symbols[ith, ::sps].imag.min()/4 ,
                         self.equalized_symbols[ith, ::sps].imag.max() + self.equalized_symbols[ith, ::sps].imag.max()/4])
            ax.set_title('scatterplot after Equalizer')
        
        plt.tight_layout()
        if self.backend == 'mpl':
            plt.show()
        else:
            import visdom
            vis = visdom.Visdom(env='receiver_dsp')
            vis.matplot(fig)

    def plot_error(self):
        fignumber = self.equalized_symbols.shape[0]
        fig, axes = plt.subplots(figsize=(8, 4), nrows=1, ncols=fignumber)
        for ith, ax in enumerate(axes):
            ax.plot(self.error_xpol_array[0], c='b', lw=1)
        fig.suptitle("Error Curve of the equalizer")
        plt.tight_layout()
        if self.backend =='mpl':
            plt.show()
        else:
            import visdom
            vis = visdom.Visdom(env='receiver_dsp')
            vis.matplot(fig)

    def plot_freq_response(self):
        from scipy.fftpack import fft, fftshift
        freq_res = fftshift(fft(self.wxx)), fftshift(fft(self.wxy)), fftshift(fft(self.wyx)), fftshift(fft(self.wyy))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2)
        for idx, row in enumerate(axes.flatten()):
            row.plot(np.abs(freq_res[idx][0]))
            row.set_title(f"{['wxx', 'wxy', 'wyx', 'wyy'][idx]}")
        fig.suptitle("Freq_response of the Equazlizer")

        plt.tight_layout()
        if self.backend == 'mpl':
            plt.show()
        else:
            
            import visdom
            vis = visdom.Visdom(env='receiver_dsp')
            vis.matplot(fig)

    def freq_response(self):
        from scipy.fftpack import fft, fftshift
        freq_res = fftshift(fft(self.wxx)), fftshift(fft(self.wxy)), fftshift(fft(self.wyx)), fftshift(fft(self.wyy))
        return freq_res


class CMA(Equalizer):

    def __init__(self, ntaps, lr, loops=3,backend = 'mpl'):
        super().__init__(ntaps, lr, loops,backend)

    def equalize(self, signal):

        import numpy as np

        samples_xpol = _segment_axis(signal[0], self.ntaps, self.ntaps - signal.sps)
        samples_ypol = _segment_axis(signal[1], self.ntaps, self.ntaps - signal.sps)

        self.error_xpol_array = np.zeros((self.loops, len(samples_xpol)))
        self.error_ypol_array = np.zeros((self.loops, len(samples_xpol)))

        for idx in range(self.loops):
            symbols, self.wxx, self.wxy, self.wyx, \
            self.wyy, error_xpol_array, error_ypol_array \
                = cma_equalize_core(samples_xpol, samples_ypol,
                                    self.wxx, self.wyy, self.wxy, self.wyx, self.lr)

            self.error_xpol_array[idx] = np.abs(error_xpol_array[0]) ** 2
            self.error_ypol_array[idx] = np.abs(error_ypol_array[0]) ** 2

        self.equalized_symbols = symbols
        signal.samples = symbols
        return signal


class LMS(Equalizer):

    def __init__(self,ntaps,lr,loops,train_symbols,train_time,backend='mpl'):

        super(LMS, self).__init__(ntaps,lr,loops,backend=backend)
        self.train_symbols = train_symbols
        self.train_time = train_time

    def equalize(self, signal):
        import numpy as np
        self.train_symbols = self.train_symbols[:, self.ntaps // 2 //signal.sps:]
        samples_xpol = _segment_axis(signal[0], self.ntaps, self.ntaps - signal.sps)
        samples_ypol = _segment_axis(signal[1], self.ntaps, self.ntaps - signal.sps)

        self.error_xpol_array = np.zeros((self.loops, len(samples_xpol)))
        self.error_ypol_array = np.zeros((self.loops, len(samples_xpol)))
        #ex, ey, train_symbol, wxx, wyy, wxy, wyx, mu_train, mu_dd, is_train):
        assert self.loops >=1
        for idx in range(self.loops):
            if self.train_time:
                symbols, self.wxx, self.wxy, self.wyx, \
                self.wyy, error_xpol_array, error_ypol_array \
                    = lms_equalize_core(samples_xpol, samples_ypol,self.train_symbols,
                                        self.wxx, self.wyy, self.wxy, self.wyx, self.lr[0],None,True)

                self.error_xpol_array[idx] = np.abs(error_xpol_array[0]) ** 2
                self.error_ypol_array[idx] = np.abs(error_ypol_array[0]) ** 2
                self.train_time-=1
            else:
                symbols, self.wxx, self.wxy, self.wyx, \
                self.wyy, error_xpol_array, error_ypol_array \
                    = lms_equalize_core(samples_xpol, samples_ypol, None,
                                        self.wxx, self.wyy, self.wxy, self.wyx, None,self.lr[1],False)

                self.error_xpol_array[idx] = np.abs(error_xpol_array[0]) ** 2
                self.error_ypol_array[idx] = np.abs(error_ypol_array[0]) ** 2


        self.equalized_symbols = symbols
        signal.samples = symbols
        signal.fs = signal.baudrate

        return signal


class FrequencyOffsetComp(object):

    def __init__(self,group,apply=True):
        self.freq_offset = None
        self.group = group
        self.apply = apply

    def prop(self,signal:Signal) -> Signal:
        from .dsp_tools import _segment_axis
        from .dsp_tools import get_time_vector
        length = len(signal)// self.group
        freq_offset = []

        if length * self.group != signal.shape[1]:
            import warnings
            warnings.warn("The group can not be divided into integers and some points will be discarded")

        time_vector = get_time_vector(len(signal), signal.fs)
        time_vector = np.atleast_2d(time_vector)[0]
        last_point = 0

        xpol = _segment_axis(signal[0], length, 0)
        ypol = _segment_axis(signal[1], length, 0)
        time_vector_segment = time_vector[:length]
        phase = np.zeros_like(xpol)

        for idx, (xpol_row, ypol_row) in enumerate(zip(xpol, ypol)):
            array = np.array([xpol_row, ypol_row])
            freq = find_freq_offset(array, signal.fs, fft_size=2**18)
            phase[idx] = 2 * np.pi * freq * time_vector_segment + last_point
            freq_offset.append(freq)
            last_point = phase[idx, -1]

        if self.apply:
            xpol = xpol * np.exp(-1j * phase)
            ypol = ypol * np.exp(-1j * phase)
        xpol = xpol.flatten()
        ypol = ypol.flatten()

        signal.samples = np.array([xpol,ypol])
        self.freq_offset = freq_offset
        return signal

def find_freq_offset(sig, fs,average_over_modes = True, fft_size = 2**18):
    """
    Find the frequency offset by searching in the spectrum of the signal
    raised to 4. Doing so eliminates the modulation for QPSK but the method also
    works for higher order M-QAM.
    Parameters
    ----------
        sig : array_line
            signal array with N modes
        os: int
            oversampling ratio (Samples per symbols in sig)
        average_over_modes : bool
            Using the field in all modes for estimation
        fft_size: array
            Size of FFT used to estimate. Should be power of 2, otherwise the
            next higher power of 2 will be used.
    Returns
    -------
        freq_offset : int
            found frequency offset
    """
    if not((np.log2(fft_size)%2 == 0) | (np.log2(fft_size)%2 == 1)):
        fft_size = 2**(int(np.ceil(np.log2(fft_size))))

    # Fix number of stuff
    sig = np.atleast_2d(sig)
    npols, L = sig.shape

    # Find offset for all modes
    freq_sig = np.zeros([npols,fft_size])
    for l in range(npols):
        freq_sig[l,:] = np.abs(np.fft.fft(sig[l,:]**4,fft_size))**2

    # Extract corresponding FO
    freq_offset = np.zeros([npols,1])
    freq_vector = np.fft.fftfreq(fft_size,1/fs)/4

    for k in range(npols):
        max_freq_bin = np.argmax(np.abs(freq_sig[k,:]))
       # print(max_freq_bin,end=',')
        freq_offset[k,0] = freq_vector[max_freq_bin]


    if average_over_modes:
        freq_offset = np.mean(freq_offset)

    return freq_offset


class Superscalar:

    def __init__(self,block_length,g,filter_n,delay,pilot_number):
        '''
            block_length: the block length of the cpe
            g: paramater for pll
            filter_n: the filter taps of the ml
            pillot_number: the number of pilot symbols for each row
        '''
        self.block_length = block_length
        self.block_number = None
        self.g = g
        self.filter_n = filter_n
        self.delay = 0
        self.phase_noise = []
        self.cpr_symbol = []
        self.symbol_for_snr = []
        self.pilot_number = pilot_number
        self.const = None

    def prop(self,signal:Signal)->Signal:
        self.const = signal.constl
        res,res_symbol = self.__divide_signal_into_block(signal)
        self.block_number = len(res[0])
        for row_samples,row_symbols in zip(res,res_symbol):
            phase_noise,cpr_temp_symbol,symbol_for_snr = self.__prop_one_pol(row_samples,row_symbols)
            self.cpr_symbol.append(cpr_temp_symbol)
            self.symbol_for_snr.append(symbol_for_snr)
            self.phase_noise.append(phase_noise)

        signal.samples = np.array(self.cpr_symbol)
        signal.tx_symbols = np.array(self.symbol_for_snr)
        self.cpr_symbol = np.array(self.cpr_symbol)
        self.symbol_for_snr = np.array(self.symbol_for_snr)
        return signal

    def plot_phase_noise(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i in range(len(self.phase_noise)):
            axes = fig.add_subplot(1,len(self.phase_noise),i+1)
            axes.plot(self.phase_noise[i],lw=1,c='b')
        plt.show()


    def __divide_signal_into_block(self,signal):
        from .dsp_tools import _segment_axis
        res = []
        res_symbol = []
        for row in signal[:]:
            row = _segment_axis(row,self.block_length,0)
            res.append(row)

        for row in signal.symbol:
            row = _segment_axis(row, self.block_length, 0)
            res_symbol.append(row)

        for idx in range(len(res)):
            assert res[idx].shape == res_symbol[idx].shape
        if divmod(len(res[0]),2)[1]!=0:
            for idx in range(len(res)):
                res[idx] = res[idx][:-1,:]
                res_symbol[idx] = res_symbol[idx][:-1,::]

        return res, res_symbol

    def __prop_one_pol(self, row_samples, row_symbols):
        if divmod(len(row_samples),2)[1]!=0:
            row_samples = row_samples[:-1,:]
            row_symbols = row_symbols[:-1,:]
        ori_rx = row_samples.copy()
        ori_rx = ori_rx.reshape(-1)
        row_samples[::2,:] = row_samples[::2,::-1]
        row_symbols[::2,:] = row_symbols[::2,::-1]

        phase_angle_temp = np.mean(row_samples[::2,:self.pilot_number]/row_symbols[::2,:self.pilot_number],axis=-1,keepdims=True) \
                    + np.mean(row_samples[1::2,:self.pilot_number]/row_symbols[1::2,:self.pilot_number],axis=-1,keepdims=True)

        phase_angle_temp = np.angle(phase_angle_temp)
        # print(phase_angle_temp.shape)
        phase_angle = np.zeros((len(row_samples),1))
        phase_angle[::2] = phase_angle_temp
        phase_angle[1::2] = phase_angle_temp

        row_samples = row_samples * np.exp(-1j * phase_angle)

        cpr_symbols = self.parallel_pll(row_samples)

        cpr_symbols[::2,:] = cpr_symbols[::2,::-1]
        cpr_symbols.shape = 1,-1
        cpr_symbols = cpr_symbols[0]

        row_symbols[::2,:] = row_symbols[::2,::-1]
        row_symbols = row_symbols.reshape(-1)

        phase_noise = self.ml(cpr_symbols,ori_rx)
        # self.phase_noise = phase_angle
        # self.cpr = row_symbols * np.exp(-1j*self.phase_noise)


        return phase_noise,ori_rx * np.exp(-1j*phase_noise),row_symbols

    def ml(self,cpr,row_samples):
        from scipy.signal import lfilter
        decision_symbol = decision(cpr,self.const)
        h = row_samples/decision_symbol
        b = np.ones(2*self.filter_n + 1)
        h = lfilter(b,1,h,axis=-1)
        h = np.roll(h,-self.filter_n)
        phase = np.angle(h)
        return phase[0]


    def parallel_pll(self,samples):

        decision_symbols = samples
        cpr_symbols = samples.copy()
        phase = np.zeros(samples.shape)
        for ith_symbol in range(0,self.block_length-1):
            decision_symbols[:,ith_symbol] = decision(cpr_symbols[:,ith_symbol],self.const)
            tmp = cpr_symbols[:,ith_symbol]*np.conj(decision_symbols[:,ith_symbol])
            error = np.imag(tmp)
            phase[:,ith_symbol+1] = self.g * error + phase[:,ith_symbol]
            cpr_symbols[:,ith_symbol + 1]  = samples[:,ith_symbol + 1] * np.exp(-1j * phase[:,ith_symbol+1])

        return cpr_symbols


def decision(decision_symbols,const):
    decision_symbols = np.atleast_2d(decision_symbols)
    const = np.atleast_2d(const)[0]
    res = np.zeros_like(decision_symbols,dtype=np.complex128)
    for row_index,row in enumerate(decision_symbols):
        for index,symbol in enumerate(row):
            index_min = np.argmin(np.abs(symbol - const))
            res[row_index,index] = const[index_min]
    return res


def syncsignal(symbol_tx, rx_signal, sps, visable=False):
    '''
        :param symbol_tx: 发送符号
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果
        # 不会改变原信号
    '''
    from scipy.signal import correlate
    symbol_tx = np.atleast_2d(symbol_tx)
    sample_rx = np.atleast_2d(rx_signal[:])
    out = np.zeros_like(sample_rx)
    corr_res = []
    # assert sample_rx.ndim == 1
    # assert symbol_tx.ndim == 1
    assert sample_rx.shape[1] >= symbol_tx.shape[1]
    for i in range(symbol_tx.shape[0]):
        symbol_tx_temp = symbol_tx[i, :]
        sample_rx_temp = sample_rx[i, :]

        res = correlate(sample_rx_temp[::sps], symbol_tx_temp)
        if visable:
            plt.figure()
            plt.plot(np.abs(np.atleast_2d(res)[0]))
            plt.show()
        index = np.argmax(np.abs(res))

        corr_res.append(res)
        out[i] = np.roll(sample_rx_temp, sps * (-index - 1 + symbol_tx_temp.shape[0]))
    if isinstance(rx_signal,Signal):
        rx_signal.samples = out
        return rx_signal
    else:
        return out,corr_res


def syncsignal_tx2rx(symbol_rx, symbol_tx):
    from scipy.signal import correlate

    symbol_tx = np.atleast_2d(symbol_tx)
    symbol_rx = np.atleast_2d(symbol_rx)
    out = np.zeros_like(symbol_tx)
    # assert sample_rx.ndim == 1
    # assert symbol_tx.ndim == 1
    assert symbol_tx.shape[1] >= symbol_rx.shape[1]
    for i in range(symbol_tx.shape[0]):
        symbol_tx_temp = symbol_tx[i, :]
        sample_rx_temp = symbol_rx[i, :]

        res = correlate(symbol_tx_temp, sample_rx_temp)
        #plt.plot(np.abs(res))
        index = np.argmax(np.abs(res))

        out[i] = np.roll(symbol_tx_temp, -index - 1 + sample_rx_temp.shape[0])
    return out


def remove_dc(signal):
    samples = signal[:]
    samples = np.atleast_2d(samples)
    samples[:] = samples - np.mean(samples,axis=1,keepdims=True)
    return signal

def orthonormalize_signal(E:Signal, os=1)->Signal:
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].
    Parameters
    ----------
    E : array_like
       input signal
    os : int, optional
        oversampling ratio of the signal
    Returns
    -------
    E_out : array_likeE
        orthonormalized signal
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """
    signal = E
    E = np.atleast_2d(E[:])
    E_out = np.empty_like(E)
    for l in range(E.shape[0]):
        # Center
        real_out = E[l,:].real - E[l,:].real.mean()
        tmp_imag = E[l,:].imag - E[l,:].imag.mean()

        # Calculate scalar products
        mean_pow_inphase = np.mean(real_out**2)
        mean_pow_quadphase = np.mean(tmp_imag**2)
        mean_pow_imb = np.mean(real_out*tmp_imag)

        # Output, Imag orthogonal to Real part of signal
        sig_out = real_out / np.sqrt(mean_pow_inphase) +\
                                    1j*(tmp_imag - mean_pow_imb * real_out / mean_pow_inphase) / np.sqrt(mean_pow_quadphase)
        # Final total normalization to ensure IQ-power equals 1
        E_out[l,:] = sig_out - np.mean(sig_out[::os])
        E_out[l,:] = E_out[l,:] / np.sqrt(np.mean(np.abs(E_out[l,::os])**2))
    signal.samples = E_out
    return signal

