import numpy as np
import matplotlib.pylab as plot
from matplotlib import *
from scipy.fftpack import fft
import pylab as pl

from scipy.io import wavfile

sampleFreq, snd = wavfile.read('tone.wav')


snd = snd / (2.**15)
#print(snd)

q = snd.shape
duration  = q[0] / sampleFreq
a = q[0]
s1 = snd[:,0]
#print(s1)

timeArray = np.arange(0, a, 1)
timeArray = timeArray / sampleFreq
timeArray = timeArray * 1000  #scale to milliseconds

#plot.plot(timeArray, s1, color='k')
#plot.ylabel('Amplitude')
#plot.xlabel('Time (ms)')

n = len(s1) 
p = fft(s1) # take the fourier transform 

nUniquePts = int(np.ceil((n+1)/2.0))
p = p[0:nUniquePts]
p = abs(p)

p = p / float(n) # scale by the number of points   
p = p**2  # square it to get the power 

if n % 2 > 0: # odd number of points fft
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) -1] = p[1:len(p) - 1] * 2 # even number of points fft

freqArray = np.arange(0, nUniquePts, 1.0) * (sampleFreq / n);

plot.plot(freqArray/1000, 10*np.log10(p), color='k')
plot.xlabel('Frequency (kHz)')
plot.ylabel('Power (dB)')
