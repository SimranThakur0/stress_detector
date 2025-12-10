# src/fusion/stress_index.py
import numpy as np
from collections import deque


class StressIndex:
    """
    StressIndex: combines heart rate (BPM) and optional blink information
    into a smooth 0–100 stress score.

    - Works with:
        compute(bpm)                  # HR only
        compute(bpm, blink_count=..)  # HR + blink
        compute(bpm, blink_count)     # positional second arg

    - Internally smooths the score over the last N calls.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        weight_hr: float = 0.8,
        weight_blink: float = 0.2,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.weight_hr = weight_hr
        self.weight_blink = weight_blink

        # rolling history for smoothing
        self._history = deque(maxlen=smoothing_window)

        # nominal physiological ranges
        self.rest_hr = 60.0    # typical resting BPM
        self.high_hr = 120.0   # high-stress BPM
        self.max_hr = 160.0    # clamp upper bound

    # ---------- Normalization helpers ----------

    def _norm_hr(self, bpm) -> float:
        """Normalize BPM into [0,1] stress contribution."""
        if bpm is None:
            return 0.0

        bpm = float(bpm)
        # clamp to reasonable range
        bpm = max(40.0, min(self.max_hr, bpm))

        # linear scale: rest_hr -> 0, high_hr -> 1
        hr_norm = (bpm - self.rest_hr) / (self.high_hr - self.rest_hr)
        return float(np.clip(hr_norm, 0.0, 1.0))

    def _norm_blink(self, blink_rate) -> float:
        """
        Normalize blink rate into [0,1].

        We assume blink_rate is roughly "blinks per second" over the
        last short window:
            - ~0.2 /s  (12 per min) → relaxed
            - ~0.7 /s  (42 per min) → elevated stress
        """
        if blink_rate is None:
            return 0.0

        rate = float(blink_rate)
        blink_norm = (rate - 0.2) / (0.7 - 0.2)
        return float(np.clip(blink_norm, 0.0, 1.0))

    # ---------- Public API ----------

    def compute(self, bpm, blink_count=None) -> float:
        """
        Compute stress index in [0, 100].

        Parameters
        ----------
        bpm : float
            Estimated heart rate (BPM) from rPPG.
        blink_count : float | int | None, optional
            Blink rate proxy. Can be:
              - blinks per second over the last second/window
              - or just a small integer count (it will still scale reasonably)

        Returns
        -------
        stress_score : float
            Smoothed stress score between 0 and 100.
        """
        # 1) components
        hr_component = self._norm_hr(bpm)
        blink_component = self._norm_blink(blink_count)

        # 2) weighted fusion
        stress_raw = (
            self.weight_hr * hr_component
            + self.weight_blink * blink_component
        )

        # 3) convert to 0–100 scale
        stress_score = stress_raw * 100.0

        # 4) temporal smoothing
        self._history.append(stress_score)
        smoothed = float(np.mean(self._history))

        # final clamp
        return max(0.0, min(100.0, smoothed))
