from class_define import Signal
from dsp.dsp import remove_dc
from dsp.dsp import orthonormalize_signal
from dsp.dsp import cd_compensation
from dsp.dsp import FrequencyOffsetComp
from class_define import Fiber
from tools.tools import find_each_section

class CoherentReceiver(object):

    def __init__(self, baudrate, adc_rate, samples, tx_symbols, tx_length, **param_dict):
        '''
        :param baudrate: The baudrate of the received signal [hz]
        :param adc_rate: The sampling rate of the scope in [hz]
        :param samples:  The received samples of the signal
        :param tx_length: The propagation length of the signal in KM
        :param param_dict:
            :key: ntaps: The ntaps for the LMS_equalizater
        '''
        signal = Signal(baudrate, samples, tx_symbols, adc_rate, wavelength=1550e-9)
        self.signal = signal
        self.span = Fiber(alpha=0.2, D=16.7, length=tx_length, reference_wavelength=1550, slope=0)
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
        signals = find_each_section(self.signal,
                                    total_length=int(np.floor(len(self.signal) / self.signal.symbol.shape[1] / 2)) - 1,
                                    symbol_length=self.signal.symbol.shape[1])
        for signal in signals:
            signal = self.__equalization(signal)
            self.dsp_processed_signals.append(signal)

    def process(self):
        self.receiver_dsp()

    def __equalization(self, signal):
        from dsp.dsp import LMS, Superscalar
        from dsp.dsp import syncsignal_tx2rx as syncsignal

        lms = LMS(ntaps=self.param['ntaps'], lr=self.param['lr'], train_time=self.param['train_time'],
                  train_symbols=signal.tx_symbols, loops=self.param['loops'])
        signal.inplace_normalise()
        signal = lms.equalize(signal)
        out = syncsignal(signal.samples, signal.tx_symbols)

        signal.tx_symbols = out
        signal.tx_symbols = signal.tx_symbols[:, :signal.shape[1]]
        cpe = Superscalar(256, 0.02, 20, 0, 4)

        signal = cpe.prop(signal)
        signal.inplace_normalise()
        return signal