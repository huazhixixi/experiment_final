import joblib
import numpy as np

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

main()