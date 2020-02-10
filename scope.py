def read_data(my_instrument, ch, datanumber):
    import time
    my_instrument.write_termination = '\n'
    my_instrument.read_terminiation = '\n'
    my_instrument.write('ACQuire:STATE STOP')
    time.sleep(2)
    my_instrument.write(f'DATA:SOURCE CH{ch}')
    my_instrument.write('DATa:ENCdg SRIbinary ')
    my_instrument.write('WFMOutpre:BYT_Nr 2')
    ymult = float(my_instrument.query('WFMOutpre:YMUlt?'))
    yzero = float(my_instrument.query("WFMOutpre:YZERO?"))
    yoff = float(my_instrument.query('WFMOutpre:YOFF?'))
    my_instrument.write('Data:START 0')
    my_instrument.write(f'Data:STOP {datanumber}')
    import numpy as np
    data2 = my_instrument.query_binary_values('CURVE?', datatype='h', container=np.array, chunk_size=1024 * 10)
    data2 = (data2 - yoff) * ymult + yzero

    return data2


def save_waveform(ch1, ch2, ch3, ch4, savedir):
    import pandas as pd
    import joblib
    dataframe = pd.DataFrame(dict(ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4))
    joblib.dump(dataframe, savedir)


import time
import pyvisa
from tqdm import tqdm_notebook

rm = pyvisa.ResourceManager()
for i in tqdm_notebook(range(100)):
    my_instrument = rm.open_resource(rm.list_resources()[0])
    if i == 0:
        print(my_instrument.query('*IDN?'))
    ch1_data = read_data(my_instrument, 1, 2e6)
    # time.sleep(1)
    ch2_data = read_data(my_instrument, 2, 2e6)
    #  time.sleep(1)
    ch3_data = read_data(my_instrument, 3, 2e6)
    # time.sleep(1)
    ch4_data = read_data(my_instrument, 4, 2e6)
    #  time.sleep(1)
    my_instrument.write('ACQuire:STATE RUN')

    save_waveform(ch1_data, ch2_data, ch3_data, ch4_data, f'f:/ai数据/ai建模数据验证/4dbm/{i}')
    time.sleep(1.5)

    my_instrument.close()
rm.close()

