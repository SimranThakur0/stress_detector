'''# src/rppg/rppg_extractor.py
import numpy as np
import time
from scipy.signal import find_peaks

from src.rppg.signal_cleaning import bandpass_filter, moving_average, zscore, detrend_signal
from src.rppg.bpm_estimator import estimate_bpm_fft, estimate_bpm_autocorr


try:
    import vhr
    HAS_PYVHR = True
except Exception:
    HAS_PYVHR = False


class RPPGExtractor:
    """
    Remote Photoplethysmography (rPPG) signal extractor with HRV support.
    
    Now extracts:
    1. Average BPM (heart rate)
    2. RR intervals (for HRV analysis)
    3. Individual heartbeat timing
    """
    
    def __init__(self, fs=30, window_size_seconds=10, region="forehead", use_pyvhr=True):
        """
        Initialize rPPG extractor.
        
        Args:
            fs: Sampling frequency (frames/sec)
            window_size_seconds: Window size for BPM estimation
            region: ROI region (e.g., 'glabella', 'forehead')
            use_pyvhr: Whether to use pyVHR library (if available)
        """
        self.fs = fs
        self.window_size = int(window_size_seconds * fs)
        self.region = region
        self.buffer = []  # RGB signal buffer
        self.timestamps = []  # Frame timestamps
        self.use_pyvhr = use_pyvhr and HAS_PYVHR

        # NEW: HRV-related storage
        self.rr_intervals = []  # RR intervals in milliseconds
        self.peak_timestamps = []  # Timestamps of detected peaks
        self.last_peak_detection = time.time()
        
        # Peak detection parameters (tuned for rPPG)
        self.min_peak_distance = int(0.4 * fs)  # Min 0.4s between peaks (150 BPM max)
        self.peak_height_threshold = 0.3  # Relative to signal std
        self.peak_prominence = 0.2  # Peak must stand out

        if self.use_pyvhr:
            pass  # pyVHR setup if needed

    def push_frame(self, roi_bgr):
        """
        Process new ROI frame and add to buffer.
        
        Args:
            roi_bgr: Cropped ROI image (H,W,3, BGR format)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            # Push NaN to keep timing consistent
            self.buffer.append(np.nan)
            self.timestamps.append(time.time())
        else:
            # Compute mean RGB values
            mean_b = np.mean(roi_bgr[:, :, 0])
            mean_g = np.mean(roi_bgr[:, :, 1])
            mean_r = np.mean(roi_bgr[:, :, 2])
            
            # CHROM-like projection for better pulse signal
            # This emphasizes pulse while reducing motion artifacts
            s = (0.77 * mean_r) - (0.51 * mean_g) + (0.14 * mean_b)
            
            self.buffer.append(s)
            self.timestamps.append(time.time())

        # Trim buffer to window size
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]

    def get_raw_signal(self):
        """
        Get raw signal and timestamps.
        
        Returns:
            tuple: (signal_array, timestamp_array)
        """
        return np.array(self.buffer, dtype=float), np.array(self.timestamps, dtype=float)

    def get_clean_signal(self, low_hz=0.7, high_hz=4.0, ma_window=5):
        """
        Get cleaned and filtered signal.
        
        Args:
            low_hz: Lower cutoff frequency (default 0.7 Hz = 42 BPM)
            high_hz: Upper cutoff frequency (default 4.0 Hz = 240 BPM)
            ma_window: Moving average window size
            
        Returns:
            tuple: (cleaned_signal, timestamps)
        """
        sig, ts = self.get_raw_signal()
        
        # Handle NaNs by interpolation
        if np.isnan(sig).any():
            nans = np.isnan(sig)
            notnans = ~nans
            if notnans.any():
                sig[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(notnans), sig[notnans])
            else:
                sig = np.zeros_like(sig)

        if len(sig) < 3:
            return sig, ts

        # Signal cleaning pipeline
        sig = detrend_signal(sig)
        sig = bandpass_filter(sig, low_hz, high_hz, fs=self.fs, order=4)
        sig = moving_average(sig, window_len=ma_window)
        sig = zscore(sig)
        
        return sig, ts

    def estimate_bpm(self, low_hz=0.7, high_hz=4.0):
        """
        Estimate average BPM from signal.
        
        Returns:
            float: Estimated BPM or None if insufficient data
        """
        sig, ts = self.get_clean_signal(low_hz, high_hz)
        if len(sig) < max(4, int(1.5 * self.fs)):
            return None
        
        bpm = estimate_bpm_fft(sig, fs=self.fs, low_hz=low_hz, high_hz=high_hz)
        if bpm is None:
            bpm = estimate_bpm_autocorr(sig, fs=self.fs)
        
        return bpm

    # ==================== HRV EXTRACTION METHODS ====================

    def detect_peaks_and_rr(self, min_quality=0.5):
        """
        Detect peaks in cleaned signal and extract RR intervals.
        
        This is the KEY method for HRV analysis - it finds individual heartbeats
        in the rPPG signal and measures the time between them.
        
        Args:
            min_quality: Minimum signal quality threshold (0-1)
            
        Returns:
            list: RR intervals in milliseconds, or empty list if insufficient data
        """
        sig, ts = self.get_clean_signal()
        
        # Need at least 2 seconds of data for peak detection
        if len(sig) < int(2 * self.fs):
            return []
        
        # Check signal quality (std should be reasonable)
        sig_std = np.std(sig)
        if sig_std < 0.1:  # Signal too flat
            return []
        
        # Normalize signal for consistent peak detection
        sig_norm = (sig - np.mean(sig)) / (sig_std + 1e-8)
        
        # Adaptive peak detection parameters based on signal quality
        # Higher quality = stricter thresholds
        height = self.peak_height_threshold * (1 + min_quality)
        prominence = self.peak_prominence * (1 + min_quality)
        
        # Find peaks using scipy
        peaks, properties = find_peaks(
            sig_norm,
            distance=self.min_peak_distance,
            height=height,
            prominence=prominence
        )
        
        if len(peaks) < 2:
            return []
        
        # Convert peak indices to timestamps
        peak_times = ts[peaks]
        
        # Calculate RR intervals (time between consecutive peaks)
        rr_intervals_sec = np.diff(peak_times)  # In seconds
        rr_intervals_ms = rr_intervals_sec * 1000  # Convert to milliseconds
        
        # Filter physiologically impossible RR intervals
        # Normal heart rate: 40-200 BPM → RR: 300-1500ms
        # Allow wider range for robustness: 250-2500ms
        valid_mask = (rr_intervals_ms >= 250) & (rr_intervals_ms <= 2500)
        valid_rr = rr_intervals_ms[valid_mask]
        
        # Additional outlier filtering using IQR method
        if len(valid_rr) > 4:
            q1 = np.percentile(valid_rr, 25)
            q3 = np.percentile(valid_rr, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (valid_rr >= lower_bound) & (valid_rr <= upper_bound)
            valid_rr = valid_rr[outlier_mask]
        
        # Store for later retrieval
        self.peak_timestamps = peak_times
        self.rr_intervals = list(valid_rr)
        self.last_peak_detection = time.time()
        
        return list(valid_rr)

    def get_hrv_data(self, update=True):
        """
        Get current RR intervals for HRV analysis.
        
        Args:
            update: Whether to re-detect peaks (default True)
            
        Returns:
            dict: Contains RR intervals, peak info, and quality metrics
        """
        if update:
            rr = self.detect_peaks_and_rr()
        else:
            rr = self.rr_intervals
        
        # Calculate data quality metrics
        num_intervals = len(rr)
        has_sufficient_data = num_intervals >= 10
        has_good_quality = num_intervals >= 30
        has_excellent_quality = num_intervals >= 60
        
        # Calculate coefficient of variation (CV) as quality indicator
        if num_intervals > 0:
            rr_cv = (np.std(rr) / np.mean(rr)) * 100
            # Good HRV typically has CV between 2-15%
            quality_score = np.clip((15 - abs(rr_cv - 8)) / 15, 0, 1)
        else:
            quality_score = 0.0
        
        return {
            'rr_intervals': rr,
            'num_intervals': num_intervals,
            'num_peaks': num_intervals + 1,
            'peak_timestamps': self.peak_timestamps,
            'data_sufficient': has_sufficient_data,
            'data_quality_good': has_good_quality,
            'data_quality_excellent': has_excellent_quality,
            'quality_score': quality_score,
            'time_since_detection': time.time() - self.last_peak_detection
        }

    def get_instantaneous_hr(self):
        """
        Get instantaneous heart rate from most recent RR intervals.
        
        This is different from average BPM - it shows beat-to-beat HR changes.
        
        Returns:
            list: Instantaneous HR values in BPM, or empty list
        """
        if len(self.rr_intervals) == 0:
            return []
        
        # Convert RR intervals (ms) to instantaneous HR (BPM)
        # HR = 60000 / RR_interval_ms
        inst_hr = [60000 / rr for rr in self.rr_intervals if rr > 0]
        
        return inst_hr

    def get_signal_quality(self):
        """
        Assess current signal quality for rPPG/HRV extraction.
        
        Returns:
            dict: Quality metrics including SNR estimate and recommendations
        """
        sig, _ = self.get_clean_signal()
        
        if len(sig) < 10:
            return {
                'quality': 'INSUFFICIENT',
                'score': 0.0,
                'message': 'Need more data'
            }
        
        # Calculate signal-to-noise ratio estimate
        sig_power = np.var(sig)
        
        # Estimate noise from high-frequency components
        sig_diff = np.diff(sig)
        noise_power = np.var(sig_diff)
        
        snr = 10 * np.log10(sig_power / (noise_power + 1e-10))
        
        # Quality scoring
        if snr > 10:
            quality = 'EXCELLENT'
            score = 1.0
            message = 'High quality signal, HRV reliable'
        elif snr > 5:
            quality = 'GOOD'
            score = 0.8
            message = 'Good signal, HRV should be reliable'
        elif snr > 0:
            quality = 'FAIR'
            score = 0.5
            message = 'Fair signal, HRV may have errors'
        else:
            quality = 'POOR'
            score = 0.2
            message = 'Poor signal, improve lighting/reduce motion'
        
        return {
            'quality': quality,
            'score': score,
            'snr_db': snr,
            'message': message,
            'signal_length': len(sig),
            'signal_std': np.std(sig)
        }

    def reset(self):
        """Reset all buffers and HRV data."""
        self.buffer.clear()
        self.timestamps.clear()
        self.rr_intervals.clear()
        self.peak_timestamps.clear()
        self.last_peak_detection = time.time()

'''
"""
Enhanced rPPG Extractor with signal quality filtering and motion artifact rejection.
This version should be used instead of the original hrv_rppg_extractor.py
"""

import numpy as np
from collections import deque
from scipy import signal as sp_signal
import time


class EnhancedRPPGExtractor:
    """
    Enhanced rPPG extractor with built-in quality assessment and artifact rejection.
    """
    
    def __init__(
        self,
        fs=30,
        window_size_seconds=12,
        region='glabella',
        use_pyvhr=False,
        quality_threshold=0.4,
        motion_threshold=0.3
    ):
        """
        Initialize enhanced rPPG extractor.
        
        Args:
            fs: Sampling frequency (frames per second)
            window_size_seconds: Window size for analysis
            region: ROI region name
            use_pyvhr: Use PyVHR library (if available)
            quality_threshold: Minimum signal quality to accept (0-1)
            motion_threshold: Maximum motion to accept (0-1)
        """
        self.fs = fs
        self.window_size = int(fs * window_size_seconds)
        self.region = region
        self.use_pyvhr = use_pyvhr
        
        # Quality and motion thresholds
        self.quality_threshold = quality_threshold
        self.motion_threshold = motion_threshold
        
        # Signal buffers
        self.roi_buffer = deque(maxlen=self.window_size)
        self.quality_buffer = deque(maxlen=self.window_size)
        self.motion_buffer = deque(maxlen=self.window_size)
        
        # Quality tracking
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.motion_rejected = 0
        self.quality_rejected = 0
        
        # Previous frame for motion detection
        self.prev_roi = None
        
        # Signal processing state
        self.last_bpm = None
        self.bpm_history = deque(maxlen=10)
        
        print(f"✓ Enhanced rPPG initialized")
        print(f"  Quality threshold: {quality_threshold}")
        print(f"  Motion threshold: {motion_threshold}")
    
    def push_frame(self, roi_frame):
        """
        Push a new ROI frame with quality and motion assessment.
        
        Args:
            roi_frame: ROI image (numpy array) or None
            
        Returns:
            accepted: Boolean indicating if frame was accepted
        """
        if roi_frame is None:
            self.rejected_frames += 1
            return False
        
        # 1. Detect motion
        motion_score = self._detect_motion(roi_frame)
        
        # 2. Extract mean intensity
        mean_intensity = np.mean(roi_frame)
        
        # 3. Quick quality check on intensity
        intensity_quality = self._assess_intensity_quality(mean_intensity)
        
        # 4. Decide whether to accept frame
        accept_motion = motion_score < self.motion_threshold
        accept_quality = intensity_quality > 0.3  # Lower threshold for frame-level
        
        if accept_motion and accept_quality:
            # Accept frame
            self.roi_buffer.append(mean_intensity)
            self.quality_buffer.append(intensity_quality)
            self.motion_buffer.append(motion_score)
            self.accepted_frames += 1
            self.prev_roi = roi_frame
            return True
        else:
            # Reject frame
            self.rejected_frames += 1
            if not accept_motion:
                self.motion_rejected += 1
            if not accept_quality:
                self.quality_rejected += 1
            return False
    
    def _detect_motion(self, roi_frame):
        """
        Detect motion between consecutive frames.
        
        Args:
            roi_frame: Current ROI frame
            
        Returns:
            motion_score: 0-1 (1 = high motion)
        """
        if self.prev_roi is None:
            return 0.0
        
        # Calculate frame difference
        curr_mean = np.mean(roi_frame)
        prev_mean = np.mean(self.prev_roi)
        
        # Relative intensity change
        intensity_change = abs(curr_mean - prev_mean) / (prev_mean + 1e-10)
        
        # Clip to 0-1 range
        motion_score = np.clip(intensity_change * 10, 0, 1)
        
        return motion_score
    
    def _assess_intensity_quality(self, intensity):
        """
        Quick quality assessment based on intensity value.
        
        Args:
            intensity: Mean intensity value
            
        Returns:
            quality_score: 0-1
        """
        # Check if intensity is in reasonable range (not too dark, not saturated)
        if intensity < 30 or intensity > 240:
            return 0.0
        
        # Normalize: best quality around 100-200
        if 100 <= intensity <= 200:
            return 1.0
        elif intensity < 100:
            return intensity / 100.0
        else:  # intensity > 200
            return (255 - intensity) / 55.0
    
    def get_raw_signal(self):
        """
        Get raw signal from buffer.
        
        Returns:
            signal: Numpy array of signal values
            timestamps: Corresponding timestamps
        """
        signal_array = np.array(list(self.roi_buffer))
        timestamps = np.arange(len(signal_array)) / self.fs
        return signal_array, timestamps
    
    def get_clean_signal(self):
        """
        Get filtered and cleaned signal.
        
        Returns:
            clean_signal: Bandpass filtered signal
            timestamps: Corresponding timestamps
        """
        raw_sig, timestamps = self.get_raw_signal()
        
        if len(raw_sig) < 30:
            return raw_sig, timestamps
        
        # Detrend
        detrended = sp_signal.detrend(raw_sig)
        
        # Bandpass filter (0.75-3.5 Hz = 45-210 BPM)
        try:
            filtered = self._bandpass_filter(detrended, 0.75, 3.5)
            return filtered, timestamps
        except ValueError:
            # If filtering fails, return detrended signal
            return detrended, timestamps
    
    def _bandpass_filter(self, sig, low_hz, high_hz, order=4):
        """
        Apply bandpass filter to signal.
        
        Args:
            sig: Input signal
            low_hz: Low cutoff frequency
            high_hz: High cutoff frequency
            order: Filter order
            
        Returns:
            filtered: Filtered signal
        """
        nyquist = self.fs / 2.0
        low = low_hz / nyquist
        high = high_hz / nyquist
        
        b, a = sp_signal.butter(order, [low, high], btype='band')
        
        # Use minimum padlen to avoid error
        padlen = min(3 * max(len(a), len(b)), len(sig) - 1)
        
        if padlen < 1:
            return sig
        
        filtered = sp_signal.filtfilt(b, a, sig, padlen=padlen)
        return filtered
    
    def estimate_bpm(self):
        """
        Estimate BPM from signal using FFT.
        
        Returns:
            bpm: Estimated BPM or None
        """
        clean_sig, _ = self.get_clean_signal()
        
        if len(clean_sig) < 60:  # Need at least 2 seconds
            return None
        
        # FFT
        freqs = np.fft.rfftfreq(len(clean_sig), 1/self.fs)
        fft_vals = np.abs(np.fft.rfft(clean_sig))
        
        # Focus on cardiac range (0.75-3.5 Hz = 45-210 BPM)
        cardiac_mask = (freqs >= 0.75) & (freqs <= 3.5)
        cardiac_freqs = freqs[cardiac_mask]
        cardiac_fft = fft_vals[cardiac_mask]
        
        if len(cardiac_fft) == 0:
            return None
        
        # Find peak frequency
        peak_idx = np.argmax(cardiac_fft)
        peak_freq = cardiac_freqs[peak_idx]
        
        # Convert to BPM
        bpm = peak_freq * 60
        
        # Smooth with history
        self.bpm_history.append(bpm)
        if len(self.bpm_history) >= 3:
            bpm = np.median(list(self.bpm_history))
        
        self.last_bpm = bpm
        return bpm
    
    def get_signal_quality(self):
        """
        Get comprehensive signal quality assessment.
        
        Returns:
            quality_dict: Dictionary with quality metrics
        """
        clean_sig, _ = self.get_clean_signal()
        
        if len(clean_sig) < 30:
            return {
                'score': 0.0,
                'quality': 'Initializing',
                'details': 'Insufficient data'
            }
        
        # Calculate SNR
        snr_score = self._calculate_snr(clean_sig)
        
        # Calculate spectral quality
        spectral_score = self._calculate_spectral_quality(clean_sig)
        
        # Get recent motion and quality scores
        recent_motion = np.mean(list(self.motion_buffer)[-30:]) if len(self.motion_buffer) > 0 else 1.0
        recent_quality = np.mean(list(self.quality_buffer)[-30:]) if len(self.quality_buffer) > 0 else 0.0
        
        # Combined score
        combined_score = (
            0.35 * snr_score +
            0.30 * spectral_score +
            0.20 * recent_quality +
            0.15 * (1.0 - recent_motion)  # Invert motion (low motion = good)
        )
        
        # Acceptance rate
        total_frames = self.accepted_frames + self.rejected_frames
        acceptance_rate = self.accepted_frames / total_frames if total_frames > 0 else 0.0
        
        # Determine quality label
        if combined_score > 0.7:
            quality_label = 'Excellent'
        elif combined_score > 0.5:
            quality_label = 'Good'
        elif combined_score > 0.3:
            quality_label = 'Fair'
        else:
            quality_label = 'Poor'
        
        return {
            'score': combined_score,
            'quality': quality_label,
            'snr': snr_score,
            'spectral': spectral_score,
            'motion': recent_motion,
            'acceptance_rate': acceptance_rate,
            'accepted_frames': self.accepted_frames,
            'rejected_frames': self.rejected_frames,
            'motion_rejected': self.motion_rejected,
            'quality_rejected': self.quality_rejected
        }
    
    def _calculate_snr(self, signal_segment):
        """Calculate SNR using spectral method."""
        freqs, psd = sp_signal.periodogram(signal_segment, self.fs)
        
        # Signal power (cardiac band: 0.75-3.5 Hz)
        cardiac_mask = (freqs >= 0.75) & (freqs <= 3.5)
        signal_power = np.sum(psd[cardiac_mask])
        
        # Noise power (outside cardiac band)
        noise_mask = ~cardiac_mask
        noise_power = np.sum(psd[noise_mask])
        
        if noise_power == 0:
            return 1.0
        
        snr_ratio = signal_power / (noise_power + 1e-10)
        return np.clip(snr_ratio / 5.0, 0, 1)
    
    def _calculate_spectral_quality(self, signal_segment):
        """Calculate spectral concentration in cardiac band."""
        freqs, psd = sp_signal.periodogram(signal_segment, self.fs)
        
        cardiac_mask = (freqs >= 0.75) & (freqs <= 3.5)
        cardiac_power = np.sum(psd[cardiac_mask])
        total_power = np.sum(psd)
        
        if total_power == 0:
            return 0.0
        
        return np.clip(cardiac_power / total_power, 0, 1)
    
    def get_hrv_data(self, update=False):
        """
        Get HRV data (RR intervals).
        
        Args:
            update: Whether to update the analysis
            
        Returns:
            hrv_dict: Dictionary with HRV data
        """
        clean_sig, timestamps = self.get_clean_signal()
        
        if len(clean_sig) < 60:
            return {
                'data_sufficient': False,
                'num_intervals': 0,
                'rr_intervals': [],
                'reason': 'insufficient_data'
            }
        
        # Detect peaks
        rr_intervals = self._detect_rr_intervals(clean_sig, timestamps)
        
        # Filter outliers
        filtered_rr = self._filter_rr_outliers(rr_intervals)
        
        # Check if sufficient
        sufficient = len(filtered_rr) >= 10
        
        return {
            'data_sufficient': sufficient,
            'num_intervals': len(filtered_rr),
            'rr_intervals': filtered_rr,
            'raw_intervals': rr_intervals,
            'filtered_count': len(rr_intervals) - len(filtered_rr)
        }
    
    def _detect_rr_intervals(self, signal_data, timestamps):
        """
        Detect RR intervals from signal peaks.
        
        Args:
            signal_data: Clean signal
            timestamps: Time points
            
        Returns:
            rr_intervals: List of RR intervals in milliseconds
        """
        # Find peaks with adaptive threshold
        peak_threshold = np.std(signal_data) * 0.5
        min_distance = int(self.fs * 0.3)  # Minimum 0.3s between peaks (200 BPM max)
        
        peaks, _ = sp_signal.find_peaks(
            signal_data,
            height=peak_threshold,
            distance=min_distance
        )
        
        if len(peaks) < 2:
            return []
        
        # Calculate RR intervals in milliseconds
        peak_times = timestamps[peaks]
        rr_intervals = np.diff(peak_times) * 1000  # Convert to ms
        
        return rr_intervals.tolist()
    
    def _filter_rr_outliers(self, rr_intervals):
        """
        Filter physiologically impossible RR intervals.
        
        Args:
            rr_intervals: List of RR intervals in ms
            
        Returns:
            filtered: List of filtered intervals
        """
        if len(rr_intervals) < 3:
            return rr_intervals
        
        rr_array = np.array(rr_intervals)
        
        # Remove physiologically impossible values
        # Min: 300ms (200 BPM), Max: 2000ms (30 BPM)
        valid_mask = (rr_array >= 300) & (rr_array <= 2000)
        filtered = rr_array[valid_mask]
        
        if len(filtered) < 3:
            return filtered.tolist()
        
        # Remove statistical outliers (IQR method)
        q1 = np.percentile(filtered, 25)
        q3 = np.percentile(filtered, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)]
        
        return filtered.tolist()
    
    def reset(self):
        """Reset all buffers and state."""
        self.roi_buffer.clear()
        self.quality_buffer.clear()
        self.motion_buffer.clear()
        self.bpm_history.clear()
        
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.motion_rejected = 0
        self.quality_rejected = 0
        
        self.prev_roi = None
        self.last_bpm = None
        
        print("✓ Enhanced rPPG reset")
    
    def get_stats(self):
        """
        Get comprehensive statistics.
        
        Returns:
            stats_dict: Dictionary with statistics
        """
        total = self.accepted_frames + self.rejected_frames
        
        return {
            'total_frames': total,
            'accepted': self.accepted_frames,
            'rejected': self.rejected_frames,
            'motion_rejected': self.motion_rejected,
            'quality_rejected': self.quality_rejected,
            'acceptance_rate': self.accepted_frames / total if total > 0 else 0.0,
            'buffer_fill': len(self.roi_buffer) / self.window_size if self.window_size > 0 else 0.0
        }
