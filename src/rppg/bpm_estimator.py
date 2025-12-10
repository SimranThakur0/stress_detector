# src/rppg/bpm_estimator.py
import numpy as np
from scipy.signal import find_peaks

def estimate_bpm_fft(signal, fs, low_hz=0.7, high_hz=4.0):
    """
    Estimate BPM from 1D signal using FFT.
    fs: sampling frequency (frames per second)
    returns bpm (float) or None if not found
    """
    n = len(signal)
    if n < 4:
        return None

    # detrend and window
    sig = signal - np.mean(signal)
    window = np.hamming(n)
    sigw = sig * window

    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft = np.abs(np.fft.rfft(sigw))
    # focus on band
    idx = np.where((freqs >= low_hz) & (freqs <= high_hz))
    if len(idx[0]) == 0:
        return None
    freqs_band = freqs[idx]
    fft_band = fft[idx]

    peak_idx = np.argmax(fft_band)
    peak_freq = freqs_band[peak_idx]
    bpm = peak_freq * 60.0
    return float(bpm)

def estimate_bpm_autocorr(signal, fs, min_bpm=40, max_bpm=200):
    """Alternative: autocorrelation-based BPM estimation (robust to noise)."""
    x = signal - np.mean(signal)
    n = len(x)
    if n < 4:
        return None
    corr = np.correlate(x, x, mode='full')[n-1:]
    # ignore zero-lag peak
    d = np.diff(corr)
    start = np.where(d > 0)[0]
    if start.size == 0:
        return None
    start = start[0]
    corr = corr[start:]
    peaks, _ = find_peaks(corr)
    if len(peaks) == 0:
        return None
    peak = peaks[0] + start
    # period in samples
    period = peak
    if period == 0:
        return None
    freq = fs / period
    bpm = freq * 60.0
    if bpm < min_bpm or bpm > max_bpm:
        return None
    return float(bpm)
