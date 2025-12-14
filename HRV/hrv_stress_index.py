# src/fusion/stress_index.py
"""
Advanced Stress Index Calculator

Combines multiple physiological signals for comprehensive stress assessment:
- Heart Rate (HR) - Absolute level
- HR Trend - Rate of change
- Heart Rate Variability (HRV) - Autonomic balance
- Blink Rate - Secondary indicator

HRV is the gold standard for stress measurement and is weighted most heavily
when available.
"""

import numpy as np
from collections import deque


class StressIndex:
    """
    Multi-modal stress index combining HR, HRV, trends, and blink rate.
    
    Automatically adapts based on available data:
    - Basic mode: HR + blink (when HRV unavailable)
    - Standard mode: HR + HR trend + blink
    - Advanced mode: HRV + HR + HR trend + blink (most accurate)
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        weight_hrv: float = 0.50,      # HRV (gold standard)
        weight_hr_absolute: float = 0.25,  # Absolute HR
        weight_hr_trend: float = 0.15,     # HR change trend
        weight_blink: float = 0.10,        # Blink rate
    ) -> None:
        """
        Initialize advanced stress index calculator.
        
        Args:
            smoothing_window: Number of measurements to average (temporal smoothing)
            weight_hrv: Weight for HRV stress component (default 50%)
            weight_hr_absolute: Weight for absolute HR (default 25%)
            weight_hr_trend: Weight for HR trend (default 15%)
            weight_blink: Weight for blink rate (default 10%)
        """
        # Store original weights
        self.weight_hrv_base = weight_hrv
        self.weight_hr_abs_base = weight_hr_absolute
        self.weight_hr_trend_base = weight_hr_trend
        self.weight_blink_base = weight_blink
        
        # Active weights (will be normalized based on available data)
        self.weight_hrv = weight_hrv
        self.weight_hr_abs = weight_hr_absolute
        self.weight_hr_trend = weight_hr_trend
        self.weight_blink = weight_blink
        
        self.smoothing_window = smoothing_window

        # Rolling history for smoothing
        self._stress_history = deque(maxlen=smoothing_window)
        self._hr_history = deque(maxlen=30)
        
        # Baseline HR (established after first few readings)
        self._baseline_hr = None
        self._baseline_established = False
        
        # Track last HRV update
        self._last_hrv_stress = None
        self._hrv_available = False

        # Physiological ranges (based on research)
        self.rest_hr = 60.0      # Resting heart rate
        self.moderate_hr = 90.0   # Moderate activity/mild stress
        self.high_hr = 120.0      # High stress/exercise
        self.max_hr = 160.0       # Maximum (clamp)

    # ==================== HR ANALYSIS ====================

    def _norm_hr_absolute(self, bpm) -> float:
        """
        Normalize BPM into [0,1] stress contribution.
        
        Uses multi-level scaling for better sensitivity:
        - 60 BPM → 0 (resting)
        - 90 BPM → 0.4 (moderate)
        - 120+ BPM → 0.85+ (high stress)
        """
        if bpm is None:
            return 0.0

        bpm = float(bpm)
        bpm = max(40.0, min(self.max_hr, bpm))
        
        # Non-linear scaling
        if bpm <= self.rest_hr:
            return 0.0
        elif bpm <= self.moderate_hr:
            return 0.4 * (bpm - self.rest_hr) / (self.moderate_hr - self.rest_hr)
        elif bpm <= self.high_hr:
            return 0.4 + 0.45 * (bpm - self.moderate_hr) / (self.high_hr - self.moderate_hr)
        else:
            return 0.85 + 0.15 * (bpm - self.high_hr) / (self.max_hr - self.high_hr)

    def _compute_hr_trend(self) -> float:
        """
        Analyze HR trend to detect stress onset/recovery.
        
        Rising HR → stress increasing (>0.5)
        Stable HR → current state maintained (≈0.5)
        Falling HR → recovery/relaxation (<0.5)
        
        Returns:
            float: 0-1 where 0.5 is stable, >0.5 is rising, <0.5 is falling
        """
        if len(self._hr_history) < 5:
            return 0.5  # Neutral until we have enough data
        
        recent_hrs = list(self._hr_history)
        
        # Establish baseline from early readings
        if not self._baseline_established:
            if len(recent_hrs) >= 10:
                self._baseline_hr = np.median(recent_hrs[:10])
                self._baseline_established = True
            else:
                return 0.5
        
        # Compare recent average to baseline
        recent_avg = np.mean(recent_hrs[-5:])
        hr_change = recent_avg - self._baseline_hr
        
        # Look at short-term slope (last 10 readings)
        if len(recent_hrs) >= 10:
            time_points = np.arange(10)
            hr_values = recent_hrs[-10:]
            slope = np.polyfit(time_points, hr_values, 1)[0]
        else:
            slope = 0
        
        # Combine absolute change and slope
        change_component = np.clip(hr_change / 15.0, -1.0, 1.0)
        slope_component = np.clip(slope / 2.0, -1.0, 1.0)
        
        # Weighted combination
        trend = 0.7 * change_component + 0.3 * slope_component
        
        # Convert from [-1, 1] to [0, 1]
        return (trend + 1.0) / 2.0

    # ==================== BLINK ANALYSIS ====================

    def _norm_blink(self, blink_rate) -> float:
        """
        Normalize blink rate with scientific context.
        
        Research shows complex blink-stress relationship:
        - Normal: 15-20 blinks/min (0.25-0.33 /s)
        - Low: <10 blinks/min - attention/initial stress response
        - High: >30 blinks/min - sustained stress/fatigue
        """
        if blink_rate is None:
            return 0.3  # Assume normal baseline
        
        rate = float(blink_rate)
        
        if rate < 0:
            return 0.3
        
        # Convert to blinks per minute if needed
        if rate < 2.0:
            blinks_per_min = rate * 60.0
        else:
            blinks_per_min = rate
        
        # Categorize blink rates
        if blinks_per_min < 10:
            return 0.25  # Low - focus/freeze response
        elif blinks_per_min < 15:
            return 0.30  # Below normal
        elif blinks_per_min <= 25:
            return 0.35  # Normal range
        elif blinks_per_min <= 35:
            return 0.35 + 0.25 * (blinks_per_min - 25) / 10.0
        elif blinks_per_min <= 50:
            return 0.60 + 0.30 * (blinks_per_min - 35) / 15.0
        else:
            return 0.90 + 0.10 * min((blinks_per_min - 50) / 20.0, 1.0)

    # ==================== WEIGHT ADAPTATION ====================

    def _adapt_weights(self, has_hrv: bool, has_hr: bool, has_blink: bool):
        """
        Dynamically adjust component weights based on available data.
        
        When HRV is available, it gets highest priority (50%).
        When HRV is unavailable, redistribute its weight to HR components.
        """
        if has_hrv:
            # Advanced mode: HRV available (most accurate)
            self.weight_hrv = self.weight_hrv_base
            self.weight_hr_abs = self.weight_hr_abs_base
            self.weight_hr_trend = self.weight_hr_trend_base
            self.weight_blink = self.weight_blink_base
            self._hrv_available = True
        else:
            # Standard/Basic mode: No HRV
            # Redistribute HRV weight to HR components
            self.weight_hrv = 0.0
            
            if has_hr:
                # Give HRV's weight to HR absolute and trend
                self.weight_hr_abs = self.weight_hr_abs_base + (self.weight_hrv_base * 0.6)
                self.weight_hr_trend = self.weight_hr_trend_base + (self.weight_hrv_base * 0.4)
                self.weight_blink = self.weight_blink_base
            else:
                # No HR either (shouldn't happen, but handle gracefully)
                self.weight_hr_abs = 0.0
                self.weight_hr_trend = 0.0
                self.weight_blink = 1.0 if has_blink else 0.0
            
            self._hrv_available = False
        
        # Normalize weights to sum to 1
        total = self.weight_hrv + self.weight_hr_abs + self.weight_hr_trend + self.weight_blink
        if total > 0:
            self.weight_hrv /= total
            self.weight_hr_abs /= total
            self.weight_hr_trend /= total
            self.weight_blink /= total

    # ==================== PUBLIC API ====================

    def compute(self, bpm=None, blink_count=None, hrv_stress=None) -> float:
        """
        Compute comprehensive stress index (0-100).
        
        Automatically adapts based on available data:
        - If hrv_stress provided: Uses HRV as primary indicator (50%)
        - If only bpm provided: Uses HR + trend (75%) + blink (25%)
        - All parameters optional (returns 0 if nothing provided)
        
        Parameters
        ----------
        bpm : float, optional
            Estimated heart rate (BPM) from rPPG
        blink_count : float | int | None, optional
            Blink rate (blinks per second or total count)
        hrv_stress : float, optional
            Pre-calculated HRV-based stress score (0-100)
            If provided, this becomes the primary stress indicator
        
        Returns
        -------
        stress_score : float
            Smoothed stress score between 0 and 100
        """
        # Track HR history
        if bpm is not None:
            self._hr_history.append(float(bpm))
        
        # Store last HRV stress for fallback
        if hrv_stress is not None:
            self._last_hrv_stress = float(hrv_stress) / 100.0  # Convert to 0-1
        
        # Determine available data
        has_hrv = hrv_stress is not None
        has_hr = bpm is not None
        has_blink = blink_count is not None
        
        # Adapt weights based on available data
        self._adapt_weights(has_hrv, has_hr, has_blink)
        
        # Compute individual components
        
        # 1. HRV component (if available)
        if has_hrv:
            hrv_component = float(hrv_stress) / 100.0  # Normalize to 0-1
        elif self._last_hrv_stress is not None:
            # Use last known HRV stress with reduced weight
            hrv_component = self._last_hrv_stress
        else:
            hrv_component = 0.0
        
        # 2. HR absolute component
        hr_abs_component = self._norm_hr_absolute(bpm) if has_hr else 0.0
        
        # 3. HR trend component
        hr_trend_component = self._compute_hr_trend() if has_hr else 0.5
        
        # 4. Blink component
        blink_component = self._norm_blink(blink_count) if has_blink else 0.3
        
        # Weighted fusion
        stress_raw = (
            self.weight_hrv * hrv_component +
            self.weight_hr_abs * hr_abs_component +
            self.weight_hr_trend * hr_trend_component +
            self.weight_blink * blink_component
        )
        
        # Convert to 0-100 scale
        stress_score = stress_raw * 100.0
        
        # Temporal smoothing
        self._stress_history.append(stress_score)
        smoothed = float(np.mean(self._stress_history))
        
        return max(0.0, min(100.0, smoothed))

    def compute_basic(self, bpm, blink_count=None) -> float:
        """
        Compute basic stress index without HRV (backward compatible).
        
        This is the original method signature for backward compatibility.
        
        Parameters
        ----------
        bpm : float
            Heart rate in BPM
        blink_count : float | int | None, optional
            Blink rate
            
        Returns
        -------
        stress_score : float
            Stress score 0-100
        """
        return self.compute(bpm=bpm, blink_count=blink_count, hrv_stress=None)

    # ==================== STATUS & ANALYSIS ====================

    def get_stress_level(self, stress_score: float) -> tuple:
        """
        Categorize stress score into interpretable levels.
        
        Returns:
            tuple: (level_name, emoji, description)
        """
        if stress_score < 20:
            return "VERY LOW", "😌", "Relaxed"
        elif stress_score < 35:
            return "LOW", "🙂", "Calm"
        elif stress_score < 55:
            return "MODERATE", "😐", "Alert"
        elif stress_score < 75:
            return "HIGH", "😰", "Stressed"
        else:
            return "VERY HIGH", "🚨", "Critical"

    def get_mode(self) -> str:
        """
        Get current operating mode based on available data.
        
        Returns:
            str: "ADVANCED" (with HRV), "STANDARD" (HR+trend), or "BASIC" (HR only)
        """
        if self._hrv_available and self.weight_hrv > 0:
            return "ADVANCED"
        elif len(self._hr_history) >= 10:
            return "STANDARD"
        else:
            return "BASIC"

    def get_confidence(self) -> float:
        """
        Calculate confidence in stress measurement.
        
        Higher confidence when:
        - HRV data available
        - More HR history available
        - HR readings are stable (not erratic)
        
        Returns:
            float: Confidence score 0-1
        """
        confidence = 0.0
        
        # HRV available = +40% confidence
        if self._hrv_available:
            confidence += 0.4
        
        # HR data available = +30% confidence
        if len(self._hr_history) >= 5:
            data_confidence = min(len(self._hr_history) / 30.0, 1.0)
            confidence += 0.3 * data_confidence
        
        # Stable readings = +30% confidence
        if len(self._hr_history) >= 10:
            recent_hrs = list(self._hr_history)[-10:]
            hr_std = np.std(recent_hrs)
            stability_confidence = 1.0 - min(hr_std / 20.0, 1.0)
            confidence += 0.3 * stability_confidence
        
        return min(confidence, 1.0)

    def get_component_breakdown(self, bpm=None, blink_count=None, hrv_stress=None) -> dict:
        """
        Get detailed breakdown of stress components for analysis.
        
        Returns:
            dict: Individual component values and weights
        """
        has_hrv = hrv_stress is not None
        has_hr = bpm is not None
        has_blink = blink_count is not None
        
        self._adapt_weights(has_hrv, has_hr, has_blink)
        
        breakdown = {
            'mode': self.get_mode(),
            'confidence': self.get_confidence(),
            'components': {
                'hrv': {
                    'value': float(hrv_stress) if hrv_stress else None,
                    'weight': self.weight_hrv,
                    'active': has_hrv
                },
                'hr_absolute': {
                    'value': self._norm_hr_absolute(bpm) * 100 if has_hr else None,
                    'weight': self.weight_hr_abs,
                    'active': has_hr
                },
                'hr_trend': {
                    'value': self._compute_hr_trend() * 100 if has_hr else None,
                    'weight': self.weight_hr_trend,
                    'active': has_hr
                },
                'blink': {
                    'value': self._norm_blink(blink_count) * 100 if has_blink else None,
                    'weight': self.weight_blink,
                    'active': has_blink
                }
            }
        }
        
        return breakdown

    def reset(self):
        """Reset all history and baseline measurements."""
        self._stress_history.clear()
        self._hr_history.clear()
        self._baseline_hr = None
        self._baseline_established = False
        self._last_hrv_stress = None
        self._hrv_available = False

    def get_debug_info(self) -> dict:
        """
        Get detailed information for debugging and analysis.
        
        Returns:
            dict: Component values and metrics
        """
        return {
            'mode': self.get_mode(),
            'baseline_hr': self._baseline_hr,
            'current_hr': self._hr_history[-1] if self._hr_history else None,
            'hr_trend': self._compute_hr_trend(),
            'confidence': self.get_confidence(),
            'hrv_available': self._hrv_available,
            'last_hrv_stress': self._last_hrv_stress * 100 if self._last_hrv_stress else None,
            'weights': {
                'hrv': self.weight_hrv,
                'hr_abs': self.weight_hr_abs,
                'hr_trend': self.weight_hr_trend,
                'blink': self.weight_blink
            },
            'history_sizes': {
                'hr': len(self._hr_history),
                'stress': len(self._stress_history)
            }
        }