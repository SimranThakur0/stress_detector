# src/rppg/hrv_analyzer.py
"""
Heart Rate Variability (HRV) Analyzer

Calculates HRV metrics from RR intervals extracted from rPPG signals.
Based on standards from European Heart Journal (1996) and PhysioNet.

HRV is the gold standard for stress measurement as it reflects
autonomic nervous system activity.
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks


class HRVAnalyzer:
    """
    Calculate Heart Rate Variability metrics from RR intervals.
    
    HRV metrics provide more accurate stress assessment than heart rate alone:
    - Low HRV (low SDNN/RMSSD) = High stress
    - High LF/HF ratio = Sympathetic dominance (stress)
    """
    
    def __init__(self, min_rr_count=10):
        """
        Initialize HRV analyzer.
        
        Args:
            min_rr_count: Minimum RR intervals needed for reliable HRV (default: 10)
                         More is better: 30+ for good quality, 60+ for best
        """
        self.min_rr_count = min_rr_count
        
        # Normal ranges for reference
        self.normal_ranges = {
            'sdnn': (30, 100),      # ms
            'rmssd': (20, 70),      # ms
            'pnn50': (5, 30),       # %
            'lf_hf_ratio': (1.0, 2.5)
        }
    
    # ==================== TIME DOMAIN METRICS ====================
    
    def calculate_time_domain(self, rr_intervals):
        """
        Calculate time-domain HRV metrics.
        
        Time-domain metrics are simpler and more robust than frequency-domain.
        
        Args:
            rr_intervals: List/array of RR intervals in milliseconds
            
        Returns:
            dict: Time-domain HRV metrics or None if insufficient data
                - mean_rr: Average RR interval (ms)
                - mean_hr: Average heart rate (BPM)
                - sdnn: Standard deviation of RR intervals (ms)
                - rmssd: Root mean square of successive differences (ms)
                - pnn50: % of intervals differing by >50ms
        """
        if len(rr_intervals) < self.min_rr_count:
            return None
        
        rr = np.array(rr_intervals, dtype=float)
        
        # Basic statistics
        mean_rr = np.mean(rr)
        mean_hr = 60000 / mean_rr  # Convert RR (ms) to HR (BPM)
        
        # SDNN: Overall HRV measure
        # Lower = reduced autonomic function, higher stress
        sdnn = np.std(rr, ddof=1)
        
        # Calculate successive differences
        diff_rr = np.diff(rr)
        
        # RMSSD: Short-term variability (parasympathetic activity)
        # Lower = reduced parasympathetic activity, higher stress
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        # pNN50: % of successive RR intervals differing by >50ms
        # Lower = reduced variability, higher stress
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # Additional metrics
        sdsd = np.std(diff_rr, ddof=1)  # Standard deviation of successive differences
        
        return {
            'mean_rr': float(mean_rr),
            'mean_hr': float(mean_hr),
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'pnn50': float(pnn50),
            'sdsd': float(sdsd)
        }
    
    # ==================== FREQUENCY DOMAIN METRICS ====================
    
    def calculate_frequency_domain(self, rr_intervals, fs=4.0):
        """
        Calculate frequency-domain HRV metrics using spectral analysis.
        
        Frequency-domain metrics show autonomic balance:
        - LF (0.04-0.15 Hz): Mixed sympathetic/parasympathetic
        - HF (0.15-0.4 Hz): Parasympathetic (breathing-related)
        - LF/HF: Sympathetic-parasympathetic balance
        
        Args:
            rr_intervals: List/array of RR intervals in milliseconds
            fs: Sampling frequency for resampled signal (Hz), default 4
            
        Returns:
            dict: Frequency-domain HRV metrics or None if insufficient data
        """
        if len(rr_intervals) < 20:  # Need more data for frequency analysis
            return None
        
        try:
            rr = np.array(rr_intervals, dtype=float)
            
            # Create time series from cumulative RR intervals
            time_rr = np.cumsum(rr) / 1000.0  # Convert to seconds
            
            # Create uniform time base
            time_uniform = np.arange(0, time_rr[-1], 1/fs)
            
            if len(time_uniform) < 10:
                return None
            
            # Interpolate RR intervals to uniform sampling
            rr_uniform = np.interp(time_uniform, time_rr, rr)
            
            # Detrend (remove DC component and linear trend)
            rr_detrended = signal.detrend(rr_uniform)
            
            # Calculate Power Spectral Density using Welch's method
            nperseg = min(256, len(rr_detrended))
            freqs, psd = signal.welch(
                rr_detrended,
                fs=fs,
                nperseg=nperseg,
                scaling='density',
                window='hann'
            )
            
            # Define frequency bands (Hz)
            vlf_band = (0.003, 0.04)   # Very Low Frequency
            lf_band = (0.04, 0.15)     # Low Frequency
            hf_band = (0.15, 0.4)      # High Frequency
            
            # Calculate power in each band (using trapezoidal integration)
            vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
            lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
            hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
            
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
            
            total_power = vlf_power + lf_power + hf_power
            
            # LF/HF ratio: Sympathetic-parasympathetic balance
            # Higher ratio = more sympathetic activation (stress)
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
            
            # Normalized power (percentage)
            lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
            hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
            
            return {
                'vlf_power': float(vlf_power),
                'lf_power': float(lf_power),
                'hf_power': float(hf_power),
                'total_power': float(total_power),
                'lf_hf_ratio': float(lf_hf_ratio),
                'lf_norm': float(lf_norm),
                'hf_norm': float(hf_norm)
            }
            
        except Exception as e:
            print(f"Warning: Frequency domain calculation failed: {e}")
            return None
    
    # ==================== UNIFIED HRV CALCULATION ====================
    
    def calculate_all_metrics(self, rr_intervals):
        """
        Calculate all HRV metrics (time + frequency domain).
        
        Args:
            rr_intervals: List/array of RR intervals in milliseconds
            
        Returns:
            dict: All HRV metrics with data quality info, or None if insufficient data
        """
        if len(rr_intervals) < self.min_rr_count:
            return None
        
        time_domain = self.calculate_time_domain(rr_intervals)
        freq_domain = self.calculate_frequency_domain(rr_intervals)
        
        if time_domain is None:
            return None
        
        result = {
            'time_domain': time_domain,
            'frequency_domain': freq_domain,
            'data_quality': {
                'num_rr_intervals': len(rr_intervals),
                'sufficient_data': len(rr_intervals) >= self.min_rr_count,
                'good_quality': len(rr_intervals) >= 30,  # 30+ for reliable HRV
                'excellent_quality': len(rr_intervals) >= 60  # 60+ for best quality
            }
        }
        
        return result
    
    # ==================== STRESS CALCULATION FROM HRV ====================
    
    def get_stress_from_hrv(self, hrv_metrics):
        """
        Convert HRV metrics to stress score (0-100).
        
        Stress indicators:
        - Low SDNN → High stress
        - Low RMSSD → High stress (reduced parasympathetic)
        - High LF/HF → High stress (sympathetic dominance)
        
        Args:
            hrv_metrics: Output from calculate_all_metrics()
            
        Returns:
            float: Stress score 0-100, or None if insufficient data
        """
        if hrv_metrics is None:
            return None
        
        time_domain = hrv_metrics.get('time_domain', {})
        freq_domain = hrv_metrics.get('frequency_domain', {})
        
        if not time_domain:
            return None
        
        # Extract time-domain metrics
        sdnn = time_domain.get('sdnn', 50)
        rmssd = time_domain.get('rmssd', 30)
        pnn50 = time_domain.get('pnn50', 10)
        
        # Normalize SDNN (typical healthy range: 30-100ms)
        # Lower SDNN = higher stress
        sdnn_norm = np.clip((sdnn - 20) / 80, 0, 1)  # 20ms = max stress, 100ms = no stress
        sdnn_stress = 1.0 - sdnn_norm
        
        # Normalize RMSSD (typical healthy range: 20-70ms)
        # Lower RMSSD = higher stress
        rmssd_norm = np.clip((rmssd - 15) / 55, 0, 1)  # 15ms = max stress, 70ms = no stress
        rmssd_stress = 1.0 - rmssd_norm
        
        # Normalize pNN50 (typical healthy range: 5-30%)
        # Lower pNN50 = higher stress
        pnn50_norm = np.clip((pnn50 - 2) / 28, 0, 1)  # 2% = max stress, 30% = no stress
        pnn50_stress = 1.0 - pnn50_norm
        
        # Frequency-domain stress indicator
        if freq_domain:
            lf_hf = freq_domain.get('lf_hf_ratio', 1.5)
            # Normal LF/HF: 1.0-2.5, Stressed: >3.0
            lf_hf_stress = np.clip((lf_hf - 1.0) / 3.0, 0, 1)
            has_freq = True
        else:
            lf_hf_stress = 0.5  # Neutral if frequency domain unavailable
            has_freq = False
        
        # Weighted combination
        if has_freq:
            # Use all metrics
            stress_score = (
                0.35 * sdnn_stress +
                0.25 * rmssd_stress +
                0.15 * pnn50_stress +
                0.25 * lf_hf_stress
            ) * 100
        else:
            # Use only time-domain metrics
            stress_score = (
                0.45 * sdnn_stress +
                0.35 * rmssd_stress +
                0.20 * pnn50_stress
            ) * 100
        
        return float(np.clip(stress_score, 0, 100))
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def get_hrv_status(self, hrv_metrics):
        """
        Get human-readable HRV status.
        
        Returns:
            tuple: (status_level, emoji, description)
        """
        if hrv_metrics is None:
            return "UNKNOWN", "❓", "Insufficient data"
        
        stress = self.get_stress_from_hrv(hrv_metrics)
        
        if stress is None:
            return "UNKNOWN", "❓", "Insufficient data"
        
        if stress < 20:
            return "EXCELLENT", "💚", "Very low stress, high HRV"
        elif stress < 40:
            return "GOOD", "✅", "Low stress, good HRV"
        elif stress < 60:
            return "MODERATE", "⚠️", "Moderate stress, reduced HRV"
        elif stress < 80:
            return "POOR", "⚡", "High stress, low HRV"
        else:
            return "CRITICAL", "🚨", "Very high stress, very low HRV"
    
    def print_hrv_report(self, hrv_metrics):
        """Print formatted HRV analysis report."""
        if hrv_metrics is None:
            print("❌ Insufficient data for HRV analysis")
            return
        
        print("\n" + "="*50)
        print("📊 HEART RATE VARIABILITY (HRV) REPORT")
        print("="*50)
        
        # Data quality
        quality = hrv_metrics['data_quality']
        print(f"\n📈 Data Quality:")
        print(f"   RR Intervals: {quality['num_rr_intervals']}")
        print(f"   Quality: {'Excellent' if quality['excellent_quality'] else 'Good' if quality['good_quality'] else 'Acceptable'}")
        
        # Time domain
        td = hrv_metrics['time_domain']
        print(f"\n⏱️  Time Domain Metrics:")
        print(f"   Mean HR: {td['mean_hr']:.1f} BPM")
        print(f"   SDNN: {td['sdnn']:.1f} ms (Normal: 30-100 ms)")
        print(f"   RMSSD: {td['rmssd']:.1f} ms (Normal: 20-70 ms)")
        print(f"   pNN50: {td['pnn50']:.1f}% (Normal: 5-30%)")
        
        # Frequency domain
        if hrv_metrics['frequency_domain']:
            fd = hrv_metrics['frequency_domain']
            print(f"\n🌊 Frequency Domain Metrics:")
            print(f"   LF Power: {fd['lf_power']:.2f}")
            print(f"   HF Power: {fd['hf_power']:.2f}")
            print(f"   LF/HF Ratio: {fd['lf_hf_ratio']:.2f} (Normal: 1.0-2.5)")
        
        # Stress assessment
        stress = self.get_stress_from_hrv(hrv_metrics)
        status, emoji, desc = self.get_hrv_status(hrv_metrics)
        print(f"\n🎯 Stress Assessment:")
        print(f"   Stress Score: {stress:.1f}/100")
        print(f"   Status: {emoji} {status} - {desc}")
        print("="*50 + "\n")