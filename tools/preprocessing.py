import joblib
import numpy as np
import visa

import tqdm
class ScopeController():

    def __init__(self,savedir,items,sample_number):
        self.resources = self.rm.list_resources()[0]
        self.savedirv = savedir
        self.items = items
        self.sample_number = sample_number
    def get_data_from_scope(self):
        self.rm = visa.ResourceManager()
        for i in tqdm.tqdm(range(self.items)):
            my_instrument = self.rm.open_resource(rm.list_resources()[0])
            if i ==0:
                print(my_instrument.query('*IDN?'))
            ch1_data = self.read_data(my_instrument,1,self.sample_number)
            ch2_data = self.read_data(my_instrument,2,self.sample_number)
            ch3_data = self.read_data(my_instrument,3,self.sample_number)
            ch4_data = self.read_data(my_instrument,4,self.sample_number)
            my_instrument.write('ACQuire:STATE RUN')
          
            self.save_waveform(ch1_data,ch2_data,ch3_data,ch4_data)
            time.sleep(1.5)

            my_instrument.close()
        self.rm.close()

    def save_waveform(self,ch1,ch2,ch3,ch4,savedir):
        import pandas as pd
        import joblib
        dataframe = pd.DataFrame(dict(ch1=ch1,ch2=ch2,ch3=ch3,ch4=ch4))
        joblib.dump(dataframe,savedir)

    def read_data(self,my_instrument,ch,datanumber):
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
        data2 = my_instrument.query_binary_values('CURVE?',datatype='h',container=np.array,chunk_size = 1024*10)
        data2 = (data2 - yoff) * ymult  + yzero
        return data2

    
def readdata(file_name):
    data = joblib.load(file_name)

    ch1 = data.ch1.values
    ch2 = data.ch2.values
    ch3 = data.ch3.values
    ch4 = data.ch4.values

    ch1.shape = 1,-1
    ch2.shape = 1,-1
    ch3.shape = 1,-1
    ch4.shape = 1,-1
    ch2 = np.roll(ch2, -1, axis=1)
    ch4 = np.roll(ch4, 2, axis=1)

    xpol = ch1 + 1j * ch2
    ypol = ch3 + 1j * ch4
    
    return np.vstack((xpol,ypol))


def main():
    readdata('/Volumes/D/0dbm/0')

