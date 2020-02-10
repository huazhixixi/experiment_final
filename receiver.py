from class_define import Signal
from dsp.dsp import remove_dc
from dsp.dsp import orthonormalize_signal
from dsp.dsp import cd_compensation
from dsp.dsp import FrequencyOffsetComp
from class_define import Fiber
from tools.tools import find_each_section
import numpy as np
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
        self.snrs = []
        self.average_symbols_signal = None

    def average_symbols_over_section(self):
        min_index= np.argmin(self.snrs)
        negtive_index = []
        for ind,snr in enumerate(self.snrs):
            if snr < 16.7:
                negtive_index.append(ind)

        from tools.tools import average_over_symbol

        signals = []

        negtive_index.append(min_index)

        for index,signal in enumerate(self.dsp_processed_signals):
            if index not in negtive_index:
                signals.append(signal)

        self.average_symbols_signal = average_over_symbol(signals)

    def front_end(self):
        self.signal = remove_dc(self.signal)
        self.signal.inplace_normalise()

        self.signal = orthonormalize_signal(self.signal, os=1)
        self.signal.resample(new_sps=2)

    def receiver_dsp(self):
        self.signal = cd_compensation(self.span, self.signal, self.signal.fs)
        freq_comp = FrequencyOffsetComp(8, True)
        self.signal = freq_comp.prop(self.signal)
        print(freq_comp.freq_offset)
        signals = find_each_section(self.signal,
                                    total_length=int(np.floor(len(self.signal) / self.signal.symbol.shape[1] / 2)) - 1,
                                    symbol_length=self.signal.symbol.shape[1],is_visable=0)
        for signal in signals:
            signal = self.__equalization(signal)
            self.dsp_processed_signals.append(signal)

    def calc_snrs(self):
        for signal in self.dsp_processed_signals:
            noise = signal[:] - signal.symbol
            noise = np.abs(noise)**2
            noise = np.mean(noise,axis=-1)
            noise = noise.sum()
            self.snrs.append(10*np.log10((2-noise)/noise))


    def process(self):
        self.front_end()
        self.receiver_dsp()

    def __equalization(self, signal):
        from dsp.dsp import LMS, Superscalar,CMA
        from dsp.dsp import syncsignal_tx2rx as syncsignal

        lms = LMS(ntaps=self.param['ntaps'], lr=self.param['lr'],  loops=self.param['loops'],train_symbols=signal.tx_symbols,

                  backend='visdom',train_time=self.param['train_time'])
        #lms = CMA(ntaps=self.param['ntaps'], lr=self.param['lr'][0]/10, loops=self.param['loops'])
        signal.inplace_normalise()
        signal = lms.equalize(signal)
        out = syncsignal(signal.samples, signal.tx_symbols)

        signal.tx_symbols = out
        signal.tx_symbols = signal.tx_symbols[:, :signal.shape[1]]
        cpe = Superscalar(256, 0.1, 20, 0, 4)

        signal = cpe.prop(signal)
        #signal.scatterplot(1)

        signal.inplace_normalise()
        return signal

    def log(self,savedir,file_name):
        with open(savedir+'/'+file_name,'a+') as f:
            for snr in self.snrs:
                f.write(str(snr)+',')
            f.write('\n')



def demodulate_signals(dir):
    from scipy.io import loadmat
    import numpy as np
    from tools.preprocessing import readdata
    import os
    import tqdm
    names = os.listdir(dir)

    import joblib
    for name in range(211,217):

        samples = readdata(dir + '/'+ str(name))
        # samples = readdata('xixi0')
        symbols = loadmat('txqpsk.mat')['tx_symbols']
        # samples[0],samples[1] = samples[1],samples[0]
        symbols[1] = np.conj(symbols[1])
        symbols = symbols[:, :65536]
        receiver = CoherentReceiver(20e9, 100e9, samples, symbols, 320, ntaps=99, lr=[0.001], train_time=3, loops=3)
        receiver.process()
        receiver.calc_snrs()
        print(receiver.snrs)
        # if np.any(np.isnan(receiver.snrs)):
        #     continue
        # if np.all(np.array(receiver.snrs)<16.7):
        #     continue
        #
        # receiver.average_symbols_over_section()
        # if receiver.average_symbols_signal is None:
        #     continue
        # receiver.log('./log', 'logfile.txt')
        # joblib.dump(receiver, f'f:/ai数据/ai建模数据验证/3dbm_demodulated/{int(name)}')


def average_all_demoulated_signals(read_dir,thres):
    import os
    names = os.listdir(read_dir)
    names = map(lambda x:read_dir+'/'+x,names)
    import joblib
    names = list(names)
    x = 0

    cnt = 0
    for name in names:
        signal = joblib.load( name)
        symbol = signal.average_symbols_signal
        noise = symbol.samples - symbol.tx_symbols
        noise = np.mean(np.abs(noise) ** 2, axis=-1).sum()
        print(name)
        if 10*np.log10((2-noise)/noise) < thres:
            continue
        else:
            x = x + symbol.samples
            cnt+=1


    y = x/cnt
    print(len(names))
    return y,symbol.tx_symbols


if __name__ == '__main__':
    demodulate_signals('f:/ai数据/ai建模数据验证/3dbm/')
    # samples,tx_symbols = average_all_demoulated_signals('f:/ai数据/ai建模数据验证/4dbm_demodulated/',thres = 16)
    #
    # noise = samples - tx_symbols
    #
    # noise = np.mean(np.abs(noise) ** 2, axis=-1)[1]
    #
    # print(10 * np.log10((1 - noise) / noise))

