import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq

from src.landmarks.face_mesh import FaceMeshDetector
from src.rppg.roi_extractor import extract_roi_frame
from src.rppg.rppg_extractor import RPPGExtractor
from src.fusion.stress_index import StressIndex


VIDEO_PATH = "D:\\stress_detector\\eval_data\\subject1\\vid.avi"

FPS = 30
WINDOW_SECONDS = 10   


def compute_snr(signal, fs):
    sig = detrend(signal)

    freqs = rfftfreq(len(sig), 1 / fs)
    spectrum = np.abs(rfft(sig))

    hr_band = (freqs >= 0.7) & (freqs <= 3.0)

    signal_peak = np.max(spectrum[hr_band])
    noise_energy = np.mean(spectrum[~hr_band]) + 1e-6

    snr = 20 * np.log10(signal_peak / noise_energy)
    return snr, signal_peak, noise_energy


def estimate_bpm(signal, fs):
    sig = detrend(signal)
    freqs = rfftfreq(len(sig), 1 / fs)
    spectrum = np.abs(rfft(sig))

    band = (freqs >= 0.7) & (freqs <= 3.0)
    if np.sum(band) == 0:
        return None

    peak_freq = freqs[band][np.argmax(spectrum[band])]
    return peak_freq * 60.0


detector = FaceMeshDetector()
rppg = RPPGExtractor(fs=FPS, window_size_seconds=WINDOW_SECONDS, region="forehead")
stress_model = StressIndex()

print("\n=== Running UBFC-rPPG Evaluation (with Stress Index) ===")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0

pred_bpm = []
pred_stress = []
pred_time = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = detector.detect(frame)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        roi, bbox = extract_roi_frame(frame, face.landmark, w, h, region="forehead")

        if roi is not None:
            rppg.push_frame(roi)

        raw_sig, _ = rppg.get_raw_signal()

        if len(raw_sig) > FPS * 4:  # 4 seconds minimum buffer
            bpm = estimate_bpm(raw_sig, FPS)

            if bpm is not None:
                # UBFC-rPPG has no blink info, so we just pass 0.
                stress = stress_model.compute(bpm, blink_count=0)

                pred_bpm.append(bpm)
                pred_stress.append(stress)
                pred_time.append(frame_index / FPS)

    frame_index += 1

cap.release()

pred_bpm = np.array(pred_bpm)
pred_stress = np.array(pred_stress)
pred_time = np.array(pred_time)

print("\n=== Final Results ===")
if len(pred_bpm) > 0:
    print(f"Estimated BPM range: {np.min(pred_bpm):.1f}–{np.max(pred_bpm):.1f}")
else:
    print("No BPM values estimated.")

if len(pred_stress) > 0:
    print(f"Stress Index range: {np.min(pred_stress):.2f}–{np.max(pred_stress):.2f}")
else:
    print("No Stress Index values computed.")

# Compute SNR of final raw signal
raw_sig, _ = rppg.get_raw_signal()
if len(raw_sig) > 0:
    snr, sig_peak, noise = compute_snr(raw_sig, FPS)
    print(f"SNR: {snr:.2f} dB")
else:
    print("No rPPG signal available for SNR computation.")


# ============================================================
# PLOTS
# ============================================================

if len(raw_sig) > 0:
    # 1 — rPPG waveform
    plt.figure(figsize=(14, 5))
    plt.plot(raw_sig)
    plt.title("Extracted rPPG Waveform")
    plt.xlabel("Samples")
    plt.grid()

    # 2 — FFT spectrum
    freqs = rfftfreq(len(raw_sig), 1 / FPS)
    spectrum = np.abs(rfft(detrend(raw_sig)))

    plt.figure(figsize=(14, 5))
    plt.plot(freqs, spectrum)
    plt.xlim(0, 4)
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.grid()

if len(pred_time) > 0:
    # 3 — BPM over time
    plt.figure(figsize=(14, 5))
    plt.plot(pred_time, pred_bpm)
    plt.title("Estimated BPM over time")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid()

    # 4 — Stress Index over time
    plt.figure(figsize=(14, 5))
    plt.plot(pred_time, pred_stress)
    plt.title("Predicted Stress Index over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Stress Index")
    plt.grid()

plt.show()
