# src/rppg/signal_cleaning.py
import numpy as np
from scipy.signal import butter, filtfilt, detrend

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, lowcut, highcut, fs, order=4):
    if len(sig) < 3:
        return sig
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, sig, padlen=3*(max(len(a), len(b))))
    return filtered

def moving_average(sig, window_len=5):
    if window_len <= 1:
        return sig
    window = np.ones(window_len)/window_len
    return np.convolve(sig, window, mode='same')

def zscore(sig):
    s = np.array(sig, dtype=float)
    m = np.mean(s)
    sd = np.std(s)
    if sd == 0:
        return s - m
    return (s - m) / sd

def detrend_signal(sig):
    return detrend(sig)
