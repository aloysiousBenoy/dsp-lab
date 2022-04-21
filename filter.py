#6. Design of FIR filter by Hamming window

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# define window functions for filtering

def sinc_filter(M, fc):
    if M%2:
        raise Exception('M must be odd')
    return np.sinc(2*fc*(np.arange(M + 1) - M/2))

def hamming(M):
    if M%2:
        raise Exception('M must be odd')
    return 0.54 - 0.46*np.cos(2*np.pi*np.arange(M + 1)/M)



f0 = 20 #20Hz
ts = 0.01 # i.e. sampling frequency is 1/ts = 100Hz
sr = 1/ts # aka. sampling freq. 

# make the time values
x = np.arange(-10, 10, ts)

# generate the signal to be filtered
signal = (np.cos(2*np.pi*f0*x) + np.sin(2*np.pi*2*f0*x) + 
                np.cos(2*np.pi*0.5*f0*x) + np.cos(2*np.pi*1.5*f0*x))

#build filters

#Low pass
M = 100 #number of taps in filter
fc = 0.25 #i.e. normalised cutoff frequency 1/4 of sampling rate i.e. 25Hz

h = sinc_filter(M, fc)*hamming(M)
ham_lpf = h/h.sum()

# get the filtered output signal by convolving the signal with the filter
y_ham = np.convolve(signal, ham_lpf)


# Take FFT of input signal to plot the spectrum
X = fft(signal)

# below steps are for frequency 
Nx = len(X)
n = np.arange(Nx)
T = Nx/sr
freq = n/T 



plt.stem(freq, abs(X), markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude of Input |X(freq)|')
plt.show()

# To plot the spectrum of filtered signal
Y=fft(y_ham)
Ny = len(Y)
n = np.arange(Ny)
T = Ny/sr
freq = n/T 

plt.stem(freq, abs(Y), markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude of Output |Y(freq)|')
plt.show()