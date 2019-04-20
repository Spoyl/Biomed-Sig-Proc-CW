# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:30:09 2019

@author: 26793504
"""

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat

rcParams.update({'font.size': 16})
rcParams["font.family"]="Dejavu Serif"
rcParams["mathtext.fontset"]="dejavuserif"

# GLOBAL VARIABLES
FILENAME = "eeg.mat"
CATEGORY = "calibrated_sig"
N = 256
NPERSEG = 250
FC1 = 8
FC2 = 13


def power_spec(O1, O2, N, O1_filt, O2_filt, fc1, fc2):
    """
    Qbiii
    Plot the power spectra of a signal and its filtered counterpart.
    Also calculates the area within a frequency band.
    """
    
    upper = int(N/2)
    
    ps1=(np.abs(np.fft.fft(O1, N)))**2
    ps2=(np.abs(np.fft.fft(O2, N)))**2
    ps3=(np.abs(np.fft.fft(O1_filt, N)))**2
    ps4=(np.abs(np.fft.fft(O2_filt, N)))**2
    
    freq=np.fft.fftfreq(N, 1./256.)     # create frequency range for x axis
    
    intgO1_filt = np.trapz(ps3[8:13], freq[8:13])
    print(intgO1_filt)
    
    plt.figure(figsize=(16, 6))
    plt.semilogy(freq[1:upper], ps1[1:upper], label="Unfiltered")
    plt.semilogy(freq[1:upper], ps3[1:upper], label="Filtered")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.title("Power Spectrum of O1 Signal")
    plt.legend()
    plt.grid()
    
    plt.figure(figsize=(16, 6))
    plt.semilogy(freq[1:upper], ps2[1:upper], label="Unfiltered")
    plt.semilogy(freq[1:upper], ps4[1:upper], label="Filtered")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.title("Power Spectrum of O2 Signal")
    plt.legend()
    plt.grid()
    
    return intgO1_filt
    

def mean_power(sig_dat1, sig_dat2, N, O1_filt, O2_filt):
    """
    Qbii
    Find the mean squared power (MSP) for each set of signals.
    
    Returns:
        pmean1 (float): MSP O1
        pmean2 (float): MSP O2
        pmean1_filt (float): MSP O1_filtered
        pmean2_filt (float): MSP O2_filtered
    """
    
    ps1=(np.abs(np.fft.fft(sig_dat1, N)))**2    #calculate power spectra
    ps2=(np.abs(np.fft.fft(sig_dat2, N)))**2
    ps3=(np.abs(np.fft.fft(O1_filt, N)))**2
    ps4=(np.abs(np.fft.fft(O2_filt, N)))**2
    
    pMean1=np.mean(ps1)                         #find mean
    pMean2=np.mean(ps2)
    pMean1_filt=np.mean(ps3)
    pMean2_filt=np.mean(ps4)
    
    print("Mean power of the unfiltered signal:\nO1\t\tO2")
    print(str(round(pMean1, 0))+"\t"+str(round(pMean2, 0)))
    print("\nMean power of the filtered signal:\nO1\t\tO2")
    print(str(round(pMean1_filt, 0))+"\t\t"+str(round(pMean2_filt, 0)))
    
    return pMean1,pMean2,pMean1_filt,pMean2_filt
    

def apply_filter(b, a, t, O1, O2):
    """
    Apply filter to data and plot the output
    
    Returns:
        O1_filt (array): O1 data filtered
        O2_filt (array): O2 data filtered
    """
    
    O1_filt=signal.lfilter(b,a,O1)
    O2_filt=signal.lfilter(b,a,O2)
    
    plt.figure(figsize = (16, 6))
    plt.plot(t[0:300], O1_filt[0:300], label="O1")
    plt.plot(t[0:300], O2_filt[0:300], label="O2")
    plt.xlabel("Time, sec")
    plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.legend()
    plt.grid()
    plt.show()
    
    return O1_filt,O2_filt


def cheby_band(fc1, fc2, fs, order=5):
    """
    Create a chebyshev bandpass filter between the frequencies
    of fc1 and fc2. Sharpest transition.
    
    Returns:
        b (array): numerator coefficients
        a (array): denominator coefficients
    """
    
    fnyq=fs/2
    passband = [fc1/fnyq, fc2/fnyq]     #numpy requires a normalized frequency for digital signals
    
    b,a=signal.cheby1(order, 1, passband, btype="band")
    w,h=signal.freqz(b,a)
    
    plt.figure(figsize=(10,6))
    plt.semilogx((fs * 0.5 / np.pi) * w, abs(h),label=str(order))
    plt.axvline(fc1,color= "orange")
    plt.axvline(fc2, color="orange")
    plt.xlabel("Frequency")
    plt.ylabel("Gain")
    plt.title("Chebyshev Bandpass Filter")
    plt.legend(title="Order")
    plt.grid()
    plt.show()
    
    return b,a
    

def butter_band(fc1, fc2, fs, order=5):
    """
    Create a butterworth bandpass filter between the frequencies
    of fc1 and fc2. Optimal flatness in passband
    
    Returns:
        b (array): numerator coefficients
        a (array): denominator coefficients
    """
    
    fnyq=fs/2
    passband = [fc1/fnyq, fc2/fnyq]     #numpy requires a normalized (?) frequency for digital signals
    
    b,a = signal.butter(order,passband,btype='band')    #make filter
    w,h=signal.freqz(b,a)   #find frequencies
    
    plt.figure(figsize=(10,6))
    plt.semilogx((fs*0.5/np.pi)*w, abs(h), label=str(order))
    plt.axvline(fc1,color= "orange")
    plt.axvline(fc2, color="orange")
    plt.xlabel("Frequency")
    plt.ylabel("Gain")
    plt.title("Butterworth Bandpass Filter")
    plt.legend(title="Order")
    plt.grid()
    plt.show()
    
    return b,a
    

def getData(filename, category):
    """
    Get the matfile data from current directory.
    Returns:
        array: [O1, O2, fs, t] 
    """
    
    matfile = loadmat(filename)
    calibrated_sig = matfile[category]

    O2 = calibrated_sig[:,0]
    O1 = calibrated_sig[:,1]
    fs = 256
    t=np.arange(0,len(O2))/fs
    
    return [O1, O2, fs, t]  


def rm_mean(O1, O2):
    """
    Remove the mean value from the signals
    
    Returns:
        O1_no_mean (array): signal O1 - mean(O1)
        O2_no_mean (array): signal O2 - mean(O2)
    """
    
    O2_no_mean = np.zeros(len(O2))
    O2_mean = np.mean(O2)
    
    for i, val in enumerate(O2):
        O2_no_mean[i] += (val-O2_mean)
    
    O1_no_mean = np.zeros(len(O1))
    O1_mean = np.mean(O1, dtype=np.float64)
    
    for i, val in enumerate(O1):
        O1_no_mean[i] += (val-O1_mean) 
        
    print(O1_mean, O2_mean)
    return O1_no_mean, O2_no_mean


def plot_sigs(O1, O2, t):
    """
    Plots the signals as a time series.
    """
    
    plt.figure(figsize = (12, 5))
    plt.plot(t[0:1500], O1[0:1500], label="O1")
    plt.plot(t[0:1500], O2[0:1500], label="O2")
    plt.grid()
    plt.xlabel("Time, sec")
    plt.ylabel("Voltage, $\mu V$")
    plt.legend()
    plt.savefig("signals8sec.png")
    plt.show()    


def plot_psd(sig_dat1, sig_dat2, fs, nperseg=256):
    """
    Plots the power spectral density of two datasets.
    
    Returns:
        (peak_f1, peak_f2): (tuple of floats) The frequencies 
        corresponding to the peak power
    """
    bar_noverlap = nperseg*0.66
    
    f, psd_O1_han = signal.welch(sig_dat1, fs, nperseg=nperseg, window = "hanning")
    f, psd_O2_han = signal.welch(sig_dat2, fs, nperseg=nperseg, window = "hanning")
    f2, psd_O1_ham = signal.welch(sig_dat1, fs, nperseg=nperseg, window="hamming")
    f2, psd_O2_ham = signal.welch(sig_dat2, fs, nperseg=nperseg, window="hanning")
    f3, psd_O1_bar = signal.welch(sig_dat1, fs, nperseg=nperseg, window="bartlett", noverlap=bar_noverlap)
    f3, psd_O2_bar = signal.welch(sig_dat2, fs, nperseg=nperseg, window="bartlett", noverlap=bar_noverlap)

#    f2, psd_O1_ham = signal.welch(sig_dat1, fs, nperseg=nperseg, window = "hamming")
#    f2, psd_O2_ham = signal.welch(sig_dat2, fs, nperseg=nperseg, window = "hamming")
    
    plt.figure(figsize = (16, 15))
    plt.subplot(311)
    plt.semilogy(f, psd_O1_han, label="O1")
    plt.semilogy(f, psd_O2_han, label="O2")
    plt.grid()
    #plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.title("Power Spectral Density Using a Hanning Window of "+str(nperseg)+" Samples")
    plt.legend()
    
    plt.subplot(312)
    plt.semilogy(f2, psd_O1_ham, label="O1")
    plt.semilogy(f2, psd_O2_ham, label="O2")
    plt.grid()
    plt.title("Power Spectral Density Using a Hamming Window of "+str(nperseg)+" Samples")
    plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.legend()
    plt.savefig("O2_O1_PSD_plot.png")
    
    plt.subplot(313)
    plt.semilogy(f2, psd_O1_bar, label="O1")
    plt.semilogy(f2, psd_O2_bar, label="O2")
    plt.grid()
    plt.title("Power Spectral Density Using a Bartlett Window of "+str(nperseg)+" Samples")
    plt.xlabel("Frequency, $\omega$")
    #plt.ylabel("Voltage, $\mu V^2 / Hz$")
    plt.legend()
    plt.savefig("O2_O1_PSD_plot.png")
    plt.show()
    
    peak_f1=f[list(psd_O1_han).index(np.max(psd_O1_han))]
    peak_f2=f[list(psd_O2_han).index(np.max(psd_O2_han))]
    
    return peak_f1, peak_f2


def plot_window(nperseg, N):
    """
    Plots a hanning window and the corresponding FFT
    """
    
    plt.figure(figsize=(16, 6))
   
    plt.subplot(121, title="Hann Window", frame_on=False)
    window_hann = np.hanning(nperseg)
    plt.scatter(np.linspace(-nperseg/2, nperseg/2, nperseg), window_hann, marker="+", label = "Hann")
    plt.grid()
    plt.xlabel("Sample")
    plt.ylabel("Magnitude")    
    
    plt.subplot(122, title="FFT of Hann Window", frame_on=False)
    A=np.fft.fft(window_hann, N)
    amp=np.abs(A)
    amp=np.clip(amp, 0.01, 100)
    freq=np.fft.fftfreq(N)
    upper = int(N/2)
    plt.semilogy(freq[1:upper], amp[1:upper])
    plt.grid()
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude")
    plt.show()


def plot_amp_spec(sig_dat1, sig_dat2, N1, N2):
    """
    Plots the amplitude spectra of two signals using the absolute
    value of the fft as per the numpy docs.
    
    Returns:
        (peak_f1, peak_f2): (tuple of floats) The frequencies 
        corresponding to the peak amplitude
    """
    
    amp1=np.abs(np.fft.fft(sig_dat1, N))    
    freq=np.fft.fftfreq(N, 1./256.)     # find frequency range for x
        
    amp2=np.abs(np.fft.fft(sig_dat2, N))    
    upper=int(N/2)
    
    amp1N2=np.abs(np.fft.fft(sig_dat1, N2))    
    freq2=np.fft.fftfreq(N2, 1./256.)     # find frequency range for x
        
    amp2N2=np.abs(np.fft.fft(sig_dat2, N2))    
    upper2=int(N2/2)
    
    plt.figure(figsize=(16,5))  
    plt.semilogy(freq[1:upper], amp1[1:upper])
    plt.semilogy(freq[1:upper], amp2[1:upper])
    plt.grid()
    plt.xlabel("Frequency, $\omega$")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Spectrum of Occipital ECGs")    
    plt.savefig("ampspec.png")
    plt.show()

    
    peak_f1=freq[list(amp1).index(np.max(amp1[1:upper]))]
    peak_f2=freq[list(amp2).index(np.max(amp2[1:upper]))]
    
    return (peak_f1, peak_f2)


def main():
#    xf[0:int(fs)]=np.nan # set first second of data to (nan) - avoid start-up transients
    O1,O2,fs,t= getData(FILENAME, CATEGORY)
    O1, O2 = rm_mean(O1, O2)
#    b,a=butter_band(8,13,fs)
#    O1_filt,O2_filt=apply_filter(b,a,t,O1,O2)
    plot_amp_spec(O1,O2,124, 256)

if __name__ == "__main__":
    main()