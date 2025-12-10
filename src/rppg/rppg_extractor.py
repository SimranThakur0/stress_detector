# src/rppg/rppg_extractor.py
import numpy as np
import time

from .signal_cleaning import bandpass_filter, moving_average, zscore, detrend_signal
from .bpm_estimator import estimate_bpm_fft, estimate_bpm_autocorr


try:
    import vhr
    HAS_PYVHR = True
except Exception:
    HAS_PYVHR = False

class RPPGExtractor:
    def __init__(self, fs=30, window_size_seconds=10, region="forehead", use_pyvhr=True):
        """
        fs: sampling frequency (frames/sec)
        window_size_seconds: how many seconds of frames to keep for bpm estimation
        region: 'forehead' or 'left_cheek' or 'right_cheek'
        """
        self.fs = fs
        self.window_size = int(window_size_seconds * fs)
        self.region = region
        self.buffer = []  # each element: mean green value per frame (or chrom signal)
        self.timestamps = []
        self.use_pyvhr = use_pyvhr and HAS_PYVHR

        if self.use_pyvhr:
            # Example placeholder: pyVHR pipeline setup (API may vary depending on version)
            # We'll not rely on exact API; prefer to compute chrom ourselves and optionally call vhr methods if available.
            pass

    def push_frame(self, roi_bgr):
        """
        roi_bgr: cropped ROI image as numpy array (H,W,3, BGR)
        returns: None (but updates internal buffer)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            # push NaN to keep timing consistent
            self.buffer.append(np.nan)
            self.timestamps.append(time.time())
        else:
            # compute mean of green channel (most rPPG energy)
            mean_b = np.mean(roi_bgr[:, :, 0])
            mean_g = np.mean(roi_bgr[:, :, 1])
            mean_r = np.mean(roi_bgr[:, :, 2])
            # CHROM-like projection: 3*R - 2*G (simple), but we'll use just green mean as baseline
            # A simple chrominance projection:
            s = (0.77 * mean_r) - (0.51 * mean_g) + (0.14 * mean_b)  # small projection to emphasize pulse
            # We keep the green channel as primary fallback; store s for variety
            self.buffer.append(s)
            self.timestamps.append(time.time())

        # trim buffer
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]

    def get_raw_signal(self):
        return np.array(self.buffer, dtype=float), np.array(self.timestamps, dtype=float)

    def get_clean_signal(self, low_hz=0.7, high_hz=4.0, ma_window=5):
        sig, ts = self.get_raw_signal()
        # handle NaNs by interpolation
        if np.isnan(sig).any():
            nans = np.isnan(sig)
            notnans = ~nans
            if notnp := notnans.any():
                sig[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(notnans), sig[notnans])
            else:
                sig = np.zeros_like(sig)

        if len(sig) < 3:
            return sig, ts

        # detrend
        sig = detrend_signal(sig)
        # bandpass
        sig = bandpass_filter(sig, low_hz, high_hz, fs=self.fs, order=4)
        # smooth
        sig = moving_average(sig, window_len=ma_window)
        # zscore
        sig = zscore(sig)
        return sig, ts

    def estimate_bpm(self, low_hz=0.7, high_hz=4.0):
        sig, ts = self.get_clean_signal(low_hz, high_hz)
        if len(sig) < max(4, int(1.5*self.fs)):
            return None
        bpm = estimate_bpm_fft(sig, fs=self.fs, low_hz=low_hz, high_hz=high_hz)
        if bpm is None:
            bpm = estimate_bpm_autocorr(sig, fs=self.fs)
        return bpm
