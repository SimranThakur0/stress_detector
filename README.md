# Real-Time Stress Detection System

A non-contact stress detection system that combines remote photoplethysmography (rPPG) with eye blink analysis to estimate physiological stress levels in real-time using standard webcam video.

## Overview

This project implements a vision-based stress assessment pipeline that extracts heart rate through remote PPG analysis and combines it with blink detection to compute a continuous stress index score (0-100). The system runs entirely on CPU with minimal latency, suitable for real-time applications.

## Architecture

### Core Components

1. **Face Mesh Detection** (`src/landmarks/face_mesh.py`)
   - MediaPipe Face Mesh for facial landmark localization
   - 468 landmarks tracked at 30 FPS
   
2. **Eye Blink Detection** (`src/landmarks/eye_blink.py`)
   - Eye Aspect Ratio (EAR) computation
   - Blink counting based on consecutive frame threshold
   
3. **rPPG Extractor** (`src/rppg/rppg_extractor.py`)
   - Region of Interest (ROI) extraction from forehead
   - Temporal filtering and heart rate estimation
   - Signal-to-Noise Ratio (SNR) computation
   
4. **Stress Index Fusion** (`src/fusion/stress_index.py`)
   - Multi-modal signal fusion
   - Temporal smoothing for stability

## Methodology

### Signal Fusion Formula

The stress index combines two physiological markers using a weighted linear fusion:

```
stress_raw = (weight_hr × hr_normalized) + (weight_blink × blink_normalized)
stress_score = stress_raw × 100
```

**Weights:**
- Heart Rate: 0.8 (80%)
- Blink Rate: 0.2 (20%)

**Rationale:**
- Heart rate is the primary stress indicator. Elevated HR strongly correlates with sympathetic nervous system activation during stress.
- Blink rate serves as a secondary cue. Stress can manifest as increased or decreased blink rates depending on cognitive load type.
- The 80:20 weighting prioritizes the more reliable rPPG signal while incorporating blink information to capture stress patterns that HR alone might miss.

### Normalization Strategy

**Heart Rate Normalization:**
- Baseline (rest): 60 BPM → stress = 0
- Elevated (stress): 120 BPM → stress = 1
- Clamped range: 40-160 BPM

**Blink Rate Normalization:**
- Relaxed: 0.2 blinks/sec (12/min) → stress = 0
- Elevated: 0.7 blinks/sec (42/min) → stress = 1

These ranges are based on established physiological literature on autonomic nervous system responses to stress.

### Temporal Smoothing

A rolling window (5 samples) averages stress scores to reduce noise and provide stable output suitable for user-facing applications. This prevents jitter while maintaining responsiveness to genuine stress transitions.

## Model Selection

### MediaPipe Face Mesh

**Why MediaPipe:**
- Lightweight and optimized for real-time performance
- Runs efficiently on CPU without GPU acceleration
- Provides dense 468-point facial landmarks with high accuracy
- Well-maintained by Google with proven robustness across diverse conditions

**Alternative Considered:**
- Dlib: Slower inference time, requires more computational resources
- Custom CNNs: Would require extensive training data and GPU inference

### rPPG Method (Green Channel + Bandpass Filtering)

**Approach:**
- Extract forehead ROI (rich in blood vessels, minimal motion artifacts)
- Average green channel intensity across ROI pixels per frame
- Apply bandpass filter (0.7-3.0 Hz) to isolate cardiac frequency
- FFT peak detection for BPM estimation

**Why This Method:**
- No pre-trained deep learning model required, eliminating model loading latency
- Classical signal processing is deterministic and interpretable
- Sufficient accuracy for stress assessment (SNR typically >10 dB on stable video)
- Can integrate PyVHR library for advanced methods if needed (disabled by default)

**Alternative Considered:**
- Deep learning rPPG (e.g., PhysNet, DeepPhys): Higher accuracy but requires GPU and adds 50-100ms latency per inference

## Performance and Trade-offs

### Latency Management

**Key Optimizations:**

1. **Single-Pass Processing:**
   - Face detection, landmark extraction, and ROI processing happen in one forward pass
   - No separate threading needed as MediaPipe is already optimized
   
2. **Minimal Buffer Requirements:**
   - rPPG requires 1.5-4 seconds of signal for BPM estimation
   - Rolling window approach ensures continuous output without reprocessing
   
3. **Efficient Data Structures:**
   - Deques for O(1) push/pop operations in signal buffers
   - Numpy operations for vectorized computation

**Measured Performance:**
- Total pipeline latency: 30-50ms per frame on modern CPU
- Frame rate: Consistent 30 FPS on 1080p webcam input
- Memory footprint: <200 MB

### Accuracy vs Speed Trade-off

**Current Balance:**
- Prioritizes real-time responsiveness over maximum accuracy
- BPM estimation accuracy: ±5 BPM under stable conditions
- Stress index correlation with ground truth: Suitable for trend detection rather than clinical diagnosis

**Acceptable Scenarios:**
- User stress monitoring in controlled environments (office, home)
- Trend analysis over minutes rather than second-by-second precision
- Applications where latency > 100ms would degrade user experience

### Known Limitations

1. **Lighting Dependency:** rPPG requires consistent, adequate lighting
2. **Motion Sensitivity:** Head movement degrades signal quality
3. **Individual Variation:** Baseline HR varies significantly across individuals
4. **No Calibration:** System uses population averages rather than personalized baselines

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy scipy matplotlib
```

## Usage

### Real-Time Detection

```bash
python main.py
```

Press `q` to quit the webcam view. A live graph window will display after closing the camera.

### Evaluation on UBFC-rPPG Dataset

```bash
python eval_ubfc_rppg.py
```

Update `VIDEO_PATH` in the script to point to your UBFC-rPPG video file.

## Output Interpretation

**Stress Index Scale:**
- 0-30: Low stress (relaxed state)
- 30-60: Moderate stress (normal arousal)
- 60-100: High stress (elevated arousal)

**BPM Display:**
- Real-time heart rate estimation
- Updated every ~1.5 seconds as new signal data accumulates

**Blink Counter:**
- Cumulative blink count since session start
- Resets when application restarts
