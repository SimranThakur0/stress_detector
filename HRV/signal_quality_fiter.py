"""
Enhanced signal quality assessment and filtering for rPPG signals.
Implements motion artifact detection and signal quality metrics.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


class SignalQualityFilter:
    """
    Comprehensive signal quality assessment for rPPG signals.
    Detects and filters out poor quality segments affected by motion artifacts.
    """
    
    def __init__(self, fs=30):
        """
        Initialize signal quality filter.
        
        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.quality_history = []
        
    def assess_signal_quality(self, signal_segment, return_details=False):
        """
        Assess the quality of an rPPG signal segment using multiple metrics.
        
        Args:
            signal_segment: 1D array of signal values
            return_details: If True, return detailed quality metrics
            
        Returns:
            quality_score: Float between 0-1 (1 = best quality)
            quality_metrics: Dict with detailed metrics (if return_details=True)
        """
        if len(signal_segment) < 30:  # Need minimum signal length
            return 0.0 if not return_details else (0.0, {})
        
        metrics = {}
        
        # 1. SNR (Signal-to-Noise Ratio)
        snr = self._calculate_snr(signal_segment)
        metrics['snr'] = snr
        
        # 2. Spectral Quality (energy concentration in cardiac band)
        spectral_quality = self._calculate_spectral_quality(signal_segment)
        metrics['spectral_quality'] = spectral_quality
        
        # 3. Perfusion Index (signal amplitude)
        perfusion = self._calculate_perfusion_index(signal_segment)
        metrics['perfusion_index'] = perfusion
        
        # 4. Statistical Features (kurtosis and skewness)
        kurt = abs(kurtosis(signal_segment))
        skewness = abs(skew(signal_segment))
        metrics['kurtosis'] = kurt
        metrics['skewness'] = skewness
        
        # Normalize kurtosis and skewness (higher values = worse quality)
        kurt_score = np.clip(1.0 - (kurt / 10.0), 0, 1)
        skew_score = np.clip(1.0 - (skewness / 3.0), 0, 1)
        
        # 5. Zero-crossing rate (indicates noise)
        zcr = self._calculate_zero_crossing_rate(signal_segment)
        metrics['zero_crossing_rate'] = zcr
        # Normalize: ideal ZCR for 60-100 BPM at fs=30 is ~1-3 per second
        ideal_zcr = 2.0 * len(signal_segment) / self.fs
        zcr_score = np.exp(-abs(zcr - ideal_zcr) / ideal_zcr)
        
        # Combined quality score (weighted average)
        quality_score = (
            0.30 * snr +              # SNR is most important
            0.25 * spectral_quality + # Spectral concentration
            0.15 * perfusion +        # Signal amplitude
            0.15 * kurt_score +       # Statistical normality
            0.10 * skew_score +       # Statistical normality
            0.05 * zcr_score          # Regularity
        )
        
        metrics['quality_score'] = quality_score
        
        # Track quality history
        self.quality_history.append(quality_score)
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
        
        if return_details:
            return quality_score, metrics
        return quality_score
    
    def _calculate_snr(self, signal_segment):
        """
        Calculate Signal-to-Noise Ratio.
        Uses spectral method: signal power in cardiac band vs noise bands.
        """
        # Apply FFT
        freqs, psd = signal.periodogram(signal_segment, self.fs)
        
        # Define frequency bands
        cardiac_band = (0.75, 3.5)  # 45-210 BPM
        noise_bands = [(0.0, 0.5), (4.0, self.fs/2)]
        
        # Signal power (cardiac band)
        cardiac_mask = (freqs >= cardiac_band[0]) & (freqs <= cardiac_band[1])
        signal_power = np.sum(psd[cardiac_mask])
        
        # Noise power (outside cardiac band)
        noise_power = 0
        for low, high in noise_bands:
            noise_mask = (freqs >= low) & (freqs <= high)
            noise_power += np.sum(psd[noise_mask])
        
        if noise_power == 0:
            return 1.0
        
        snr_ratio = signal_power / (noise_power + 1e-10)
        # Normalize to 0-1 scale (SNR > 3 is good)
        snr_score = np.clip(snr_ratio / 5.0, 0, 1)
        
        return snr_score
    
    def _calculate_spectral_quality(self, signal_segment):
        """
        Calculate spectral quality: how concentrated the power is in cardiac band.
        """
        freqs, psd = signal.periodogram(signal_segment, self.fs)
        
        # Cardiac band (0.75-3.5 Hz = 45-210 BPM)
        cardiac_mask = (freqs >= 0.75) & (freqs <= 3.5)
        cardiac_power = np.sum(psd[cardiac_mask])
        total_power = np.sum(psd)
        
        if total_power == 0:
            return 0.0
        
        # Ratio of cardiac to total power
        concentration = cardiac_power / total_power
        return np.clip(concentration, 0, 1)
    
    def _calculate_perfusion_index(self, signal_segment):
        """
        Calculate perfusion index (signal amplitude relative to DC component).
        """
        ac_component = np.std(signal_segment)
        dc_component = np.mean(np.abs(signal_segment))
        
        if dc_component == 0:
            return 0.0
        
        perfusion = ac_component / dc_component
        # Normalize: typical good perfusion is 0.01-0.1
        perfusion_score = np.clip(perfusion / 0.1, 0, 1)
        
        return perfusion_score
    
    def _calculate_zero_crossing_rate(self, signal_segment):
        """
        Calculate zero-crossing rate (number of sign changes).
        """
        # Center signal
        centered = signal_segment - np.mean(signal_segment)
        # Count zero crossings
        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        return crossings
    
    def get_quality_trend(self, window=10):
        """
        Get recent quality trend.
        
        Returns:
            'improving', 'stable', 'degrading', or 'unknown'
        """
        if len(self.quality_history) < window * 2:
            return 'unknown'
        
        recent = np.mean(self.quality_history[-window:])
        previous = np.mean(self.quality_history[-window*2:-window])
        
        diff = recent - previous
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'degrading'
        else:
            return 'stable'


class MotionArtifactDetector:
    """
    Detect motion artifacts in video frames using multiple strategies.
    """
    
    def __init__(self, threshold=0.3):
        """
        Initialize motion artifact detector.
        
        Args:
            threshold: Motion threshold (higher = more tolerant)
        """
        self.threshold = threshold
        self.prev_roi_intensity = None
        self.motion_history = []
        
    def detect_motion(self, roi_frame):
        """
        Detect if current frame has significant motion artifacts.
        
        Args:
            roi_frame: Current ROI image (numpy array)
            
        Returns:
            has_motion: Boolean indicating if motion detected
            motion_score: Float between 0-1 (1 = high motion)
        """
        if roi_frame is None:
            return True, 1.0
        
        # Calculate mean intensity
        current_intensity = np.mean(roi_frame)
        
        if self.prev_roi_intensity is None:
            self.prev_roi_intensity = current_intensity
            return False, 0.0
        
        # 1. Intensity change detection
        intensity_change = abs(current_intensity - self.prev_roi_intensity) / (self.prev_roi_intensity + 1e-10)
        
        # 2. Calculate spatial variance (blurriness indicator)
        spatial_var = np.var(roi_frame)
        
        # Combine metrics
        motion_score = np.clip(intensity_change * 10, 0, 1)
        
        # Update history
        self.motion_history.append(motion_score)
        if len(self.motion_history) > 30:
            self.motion_history.pop(0)
        
        # Update previous
        self.prev_roi_intensity = current_intensity
        
        # Determine if motion exceeds threshold
        has_motion = motion_score > self.threshold
        
        return has_motion, motion_score
    
    def get_motion_trend(self):
        """
        Get recent motion trend.
        
        Returns:
            average motion score over recent history
        """
        if len(self.motion_history) < 10:
            return 0.0
        return np.mean(self.motion_history[-10:])
    
    def reset(self):
        """Reset detector state."""
        self.prev_roi_intensity = None
        self.motion_history = []


class RRIntervalFilter:
    """
    Filter and validate RR intervals to remove physiologically impossible values.
    """
    
    def __init__(self):
        """Initialize RR interval filter."""
        self.min_rr = 300   # 200 BPM max
        self.max_rr = 2000  # 30 BPM min
        self.max_change = 0.3  # 30% max change between consecutive RRs
        
    def filter_rr_intervals(self, rr_intervals):
        """
        Filter RR intervals to remove outliers and artifacts.
        
        Args:
            rr_intervals: List or array of RR intervals in milliseconds
            
        Returns:
            filtered_rr: Array of filtered RR intervals
            removed_count: Number of intervals removed
        """
        if len(rr_intervals) < 3:
            return np.array(rr_intervals), 0
        
        rr_array = np.array(rr_intervals)
        filtered = []
        removed_count = 0
        
        for i, rr in enumerate(rr_array):
            # 1. Check physiological range
            if rr < self.min_rr or rr > self.max_rr:
                removed_count += 1
                continue
            
            # 2. Check for sudden changes (except first interval)
            if len(filtered) > 0:
                prev_rr = filtered[-1]
                change_ratio = abs(rr - prev_rr) / prev_rr
                
                if change_ratio > self.max_change:
                    removed_count += 1
                    continue
            
            filtered.append(rr)
        
        # 3. Remove statistical outliers using IQR method
        if len(filtered) >= 5:
            q1 = np.percentile(filtered, 25)
            q3 = np.percentile(filtered, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            initial_count = len(filtered)
            filtered = [rr for rr in filtered if lower_bound <= rr <= upper_bound]
            removed_count += (initial_count - len(filtered))
        
        return np.array(filtered), removed_count
    
    def validate_hrv_quality(self, rr_intervals):
        """
        Validate if RR intervals are sufficient quality for HRV analysis.
        
        Args:
            rr_intervals: Array of RR intervals
            
        Returns:
            is_valid: Boolean
            quality_metrics: Dict with validation details
        """
        if len(rr_intervals) < 10:
            return False, {'reason': 'insufficient_data', 'count': len(rr_intervals)}
        
        # Calculate variability
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        cv = rr_std / rr_mean  # Coefficient of variation
        
        metrics = {
            'count': len(rr_intervals),
            'mean_rr': rr_mean,
            'std_rr': rr_std,
            'cv': cv
        }
        
        # Check if variability is reasonable
        # Too high CV (>0.3) suggests noise/artifacts
        # Too low CV (<0.01) suggests poor signal detection
        if cv > 0.3:
            metrics['reason'] = 'excessive_variability'
            return False, metrics
        elif cv < 0.01:
            metrics['reason'] = 'insufficient_variability'
            return False, metrics
        
        metrics['reason'] = 'valid'
        return True, metrics


# Example usage
if __name__ == "__main__":
    # Test signal quality filter
    print("Testing Signal Quality Filter...")
    
    # Generate test signal (60 BPM with some noise)
    fs = 30
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Clean signal (60 BPM = 1 Hz)
    clean_signal = np.sin(2 * np.pi * 1 * t)
    
    # Noisy signal
    noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))
    
    sqf = SignalQualityFilter(fs=fs)
    
    clean_score, clean_metrics = sqf.assess_signal_quality(clean_signal, return_details=True)
    noisy_score, noisy_metrics = sqf.assess_signal_quality(noisy_signal, return_details=True)
    
    print(f"\nClean signal quality: {clean_score:.3f}")
    print(f"  SNR: {clean_metrics['snr']:.3f}")
    print(f"  Spectral Quality: {clean_metrics['spectral_quality']:.3f}")
    
    print(f"\nNoisy signal quality: {noisy_score:.3f}")
    print(f"  SNR: {noisy_metrics['snr']:.3f}")
    print(f"  Spectral Quality: {noisy_metrics['spectral_quality']:.3f}")
    
    # Test RR interval filter
    print("\n\nTesting RR Interval Filter...")
    
    # Simulate RR intervals with outliers
    normal_rr = np.random.normal(800, 50, 20)  # 75 BPM ± variation
    outliers = np.array([200, 1500, 300])  # Physiologically impossible
    test_rr = np.concatenate([normal_rr, outliers])
    
    rr_filter = RRIntervalFilter()
    filtered_rr, removed = rr_filter.filter_rr_intervals(test_rr)
    
    print(f"Original RR intervals: {len(test_rr)}")
    print(f"Filtered RR intervals: {len(filtered_rr)}")
    print(f"Removed: {removed}")
    
    is_valid, metrics = rr_filter.validate_hrv_quality(filtered_rr)
    print(f"\nHRV Quality Valid: {is_valid}")
    print(f"Mean RR: {metrics['mean_rr']:.1f} ms")
    print(f"CV: {metrics['cv']:.3f}")

    '''

import numpy as np


def filter_rr_intervals_adaptive(rr_intervals, verbose=True):
    """
    Adaptive filtering of RR intervals with progressive strictness.
    Balances artifact removal with data retention.
    """
    if len(rr_intervals) < 3:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    original_count = len(rr_array)
    
    if verbose:
        print(f"  [Filter] Input: {len(rr_array)} intervals, "
              f"range: {rr_array.min():.0f}-{rr_array.max():.0f} ms")
    
    # Step 1: Remove physiologically impossible values
    # Relaxed bounds: 300-2000ms (30-200 BPM range)
    valid_mask = (rr_array >= 300) & (rr_array <= 2000)
    filtered = rr_array[valid_mask]
    
    if verbose:
        print(f"  [Filter] After physiological range: {len(filtered)} intervals")
    
    if len(filtered) < 3:
        return filtered.tolist()
    
    # Step 2: Adaptive outlier removal based on data quantity
    # Use IQR method with adaptive multiplier
    q1 = np.percentile(filtered, 25)
    q3 = np.percentile(filtered, 75)
    iqr = q3 - q1
    
    # Adaptive IQR multiplier: stricter when we have more data
    if len(filtered) >= 20:
        iqr_multiplier = 1.5  # Standard
    elif len(filtered) >= 10:
        iqr_multiplier = 2.0  # More lenient
    else:
        iqr_multiplier = 2.5  # Very lenient
    
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    filtered = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)]
    
    if verbose:
        print(f"  [Filter] After IQR (×{iqr_multiplier}): {len(filtered)} intervals")
    
    if len(filtered) < 3:
        return filtered.tolist()
    
    # Step 3: Stability filter (only if we have enough data)
    if len(filtered) >= 10:
        stable_filtered = [filtered[0]]
        median_rr = np.median(filtered)
        
        for i in range(1, len(filtered)):
            # Adaptive threshold based on position in array
            # More lenient at edges, stricter in middle
            if i < 3 or i > len(filtered) - 3:
                max_change = 0.40  # 40% for edge values
            else:
                max_change = 0.30  # 30% for middle values
            
            change_ratio = abs(filtered[i] - filtered[i-1]) / filtered[i-1]
            
            # Also check if value is reasonable compared to median
            deviation_from_median = abs(filtered[i] - median_rr) / median_rr
            
            if change_ratio < max_change and deviation_from_median < 0.50:
                stable_filtered.append(filtered[i])
        
        filtered = np.array(stable_filtered)
        
        if verbose:
            print(f"  [Filter] After stability check: {len(filtered)} intervals")
    
    # Step 4: MAD-based outlier removal (only if we have sufficient data)
    if len(filtered) >= 8:
        median = np.median(filtered)
        mad = np.median(np.abs(filtered - median))
        
        if mad > 0:
            # Keep values within 3.0 MAD (more lenient than before)
            filtered = filtered[np.abs(filtered - median) <= 3.0 * mad]
            
            if verbose:
                print(f"  [Filter] After MAD filter: {len(filtered)} intervals")
    
    # Final validation
    if len(filtered) > 0:
        retention_rate = (len(filtered) / original_count) * 100
        if verbose:
            print(f"  [Filter] Final: {len(filtered)} intervals "
                  f"(retained {retention_rate:.1f}%), mean: {np.mean(filtered):.0f} ms")
        
        # Quality warning if too much data was removed
        if retention_rate < 30:
            if verbose:
                print(f"  [Warning] Low retention rate - signal quality may be poor")
    
    return filtered.tolist()


def filter_rr_intervals_minimal(rr_intervals, verbose=True):
    """
    Minimal filtering - only removes clear artifacts.
    Use this if adaptive filtering is still too aggressive.
    """
    if len(rr_intervals) < 3:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    
    if verbose:
        print(f"  [Filter-Min] Input: {len(rr_array)} intervals, "
              f"range: {rr_array.min():.0f}-{rr_array.max():.0f} ms")
    
    # Only remove physiologically impossible values (wider range)
    valid_mask = (rr_array >= 250) & (rr_array <= 2500)
    filtered = rr_array[valid_mask]
    
    if verbose:
        print(f"  [Filter-Min] After range check: {len(filtered)} intervals")
    
    if len(filtered) < 3:
        return filtered.tolist()
    
    # Only remove extreme outliers (3x IQR)
    q1 = np.percentile(filtered, 25)
    q3 = np.percentile(filtered, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 3.0 * iqr
    upper_bound = q3 + 3.0 * iqr
    
    filtered = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)]
    
    if verbose:
        print(f"  [Filter-Min] Final: {len(filtered)} intervals, "
              f"mean: {np.mean(filtered):.0f} ms")
    
    return filtered.tolist()


def get_filter_quality_score(original_intervals, filtered_intervals):
    """
    Calculate a quality score based on filtering results.
    Returns score 0-100 and quality label.
    """
    if len(original_intervals) == 0:
        return 0, "No Data"
    
    retention_rate = len(filtered_intervals) / len(original_intervals)
    
    if len(filtered_intervals) < 5:
        return 20, "Poor"
    elif len(filtered_intervals) < 10:
        score = 40 + (len(filtered_intervals) - 5) * 6
        return score, "Fair"
    elif len(filtered_intervals) < 30:
        base_score = 70
        retention_bonus = min(20, retention_rate * 20)
        return base_score + retention_bonus, "Good"
    else:
        base_score = 85
        retention_bonus = min(15, retention_rate * 15)
        return base_score + retention_bonus, "Excellent"'''