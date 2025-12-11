"""
live_product_view.py

Real-time "Product View" visualization:
 - Top: webcam feed with MediaPipe face-mesh overlays + ROI rectangle
 - Bottom: two live plots:
     1) Raw rPPG signal (time-series)
     2) Stress index (0-100) over time

Requirements (already present in your project):
 - src.video.webcam_stream.WebcamStream
 - src.landmarks.face_mesh.FaceMeshDetector
 - src.rppg.rppg_extractor.RPPGExtractor
 - src.fusion.stress_index.StressIndex
 - src.landmarks.eye_blink.BlinkDetector (optional, used for blink-based stress)

Notes:
 - This script uses threading for the capture loop and matplotlib.animation.FuncAnimation
   for plotting (keeps plotting on main thread which is required by most GUI backends).
 - Configure VIDEO_FPS, PLOT_WINDOW_SECONDS and buffer lengths as needed.
 - Use Qt backend when available (matplotlib default is usually fine). If you see
   "Animation was deleted without rendering anything" warnings, try running with
   a proper Qt backend: `matplotlib.use("QtAgg")` before pyplot import.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import matplotlib
# If you have Qt installed and you experienced Matplotlib animation warnings earlier,
# uncomment the line below to force QtAgg backend. Otherwise leave default.
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your project's modules (adjust paths if needed)
from src.video.webcam_stream1 import WebcamStream
from src.landmarks.face_mesh import FaceMeshDetector
from src.landmarks.eye_blink import BlinkDetector
from src.rppg.rppg_extractor import RPPGExtractor
from src.fusion.stress_index import StressIndex
from src.rppg.roi_extractor import draw_roi

# -----------------------
# Configuration
# -----------------------
VIDEO_WINDOW_NAME = "Product View - Live Demo (press 'q' to quit)"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

VIDEO_FPS = 30                        # camera fps (used for time axis)
PLOT_WINDOW_SECONDS = 30              # length of time axis shown in plot
RPPG_SAMPLE_RATE = VIDEO_FPS          # rPPG extractor sampling rate (frames/sec)
MAX_SAMPLES = int(PLOT_WINDOW_SECONDS * RPPG_SAMPLE_RATE)

PLOT_UPDATE_INTERVAL_MS = 250         # update plots every 250 ms

# -----------------------
# Shared buffers (thread-safe with a lock)
# -----------------------
frame_lock = threading.Lock()
latest_frame = None         # BGR frame to show (with landmark overlay)
roi_bbox = None             # last detected ROI bbox for overlay

rppg_lock = threading.Lock()
rppg_buffer = deque(maxlen=MAX_SAMPLES)        # numeric rPPG samples (float)
rppg_time_buffer = deque(maxlen=MAX_SAMPLES)   # corresponding timestamps

stress_lock = threading.Lock()
stress_buffer = deque(maxlen=MAX_SAMPLES)     # stress per sample (float)
stress_time_buffer = deque(maxlen=MAX_SAMPLES) # timestamps (same as rppg_time_buffer)

bpm_lock = threading.Lock()
bpm_buffer = deque(maxlen=MAX_SAMPLES)
bpm_time_buffer = deque(maxlen=MAX_SAMPLES)

# -----------------------
# Components (instantiate)
# -----------------------
webcam = WebcamStream(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS)
face_detector = FaceMeshDetector()
blink_detector = BlinkDetector()
rppg_extractor = RPPGExtractor(fs=RPPG_SAMPLE_RATE, window_size_seconds=12, region="forehead", use_pyvhr=False)
stress_model = StressIndex(smoothing_window=5)

# flag to stop threads cleanly
running = threading.Event()
running.set()


# -----------------------
# Capture & processing thread
# -----------------------
def capture_loop():
    global latest_frame, roi_bbox
    frame_idx = 0
    start_time = time.time()

    while running.is_set():
        ret, frame = webcam.read()
        if not ret:
            # small sleep to avoid busy-loop if camera fails
            time.sleep(0.01)
            continue

        frame_h, frame_w = frame.shape[:2]
        results = face_detector.detect(frame)

        # overlay variables for visual
        overlay_frame = frame.copy()
        current_roi = None
        current_bpm = None
        current_stress = None

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # draw landmarks onto overlay_frame
            overlay_frame = face_detector.draw_landmarks(overlay_frame, results)

            # detect blink (optional) - returns EAR and blink_count (frame-level)
            ear, blink_count = blink_detector.detect_blink(face.landmark, frame_w, frame_h)

            # extract ROI (forehead by default in extractor)
            roi_img, bbox = rppg_extractor.region_from_face(frame, face.landmark, frame_w, frame_h) \
                if hasattr(rppg_extractor, "region_from_face") else (None, None)

            # fallback to older extract_roi_frame if above not present
            if roi_img is None:
                try:
                    # try to import and call your roi extractor function
                    from src.rppg.roi_extractor import extract_roi_frame
                    roi_img, bbox = extract_roi_frame(frame, face.landmark, frame_w, frame_h, region="forehead")
                except Exception:
                    roi_img, bbox = None, None

            if roi_img is not None:
                # draw roi rect
                overlay_frame = draw_roi(overlay_frame, bbox, color=(0, 200, 0), thickness=2)
                current_roi = roi_img

            # push ROI frame to rPPG pipeline (it handles None gracefully)
            rppg_extractor.push_frame(roi_img)

            # get raw rppg signal (list or numpy)
            raw_sig, _ = rppg_extractor.get_raw_signal()

            if raw_sig is not None and len(raw_sig) > 0:
                # rppg_extractor.get_raw_signal() may return list; take last sample as numeric value
                # We assume rPPGExtractor maintains a 1D numeric stream (mean green/combination) per frame
                sample = float(raw_sig[-1]) if isinstance(raw_sig, (list, tuple, np.ndarray)) else float(raw_sig)

                ts = time.time() - start_time

                # append to shared rppg buffers
                with rppg_lock:
                    rppg_buffer.append(sample)
                    rppg_time_buffer.append(ts)

                # estimate BPM using extractor's method if available (avoid heavy compute per frame)
                try:
                    bpm_val = rppg_extractor.estimate_bpm()
                except Exception:
                    bpm_val = None

                with bpm_lock:
                    if bpm_val is not None:
                        bpm_buffer.append(bpm_val)
                        bpm_time_buffer.append(ts)
                        current_bpm = bpm_val

                # compute stress from bpm and blink_count (blink_count may be zero)
                try:
                    stress_val = stress_model.compute(bpm_val, blink_count if 'blink_count' in locals() else None)
                except Exception:
                    # fallback if stress_model expects positional
                    stress_val = stress_model.compute(bpm_val)

                with stress_lock:
                    if stress_val is not None:
                        stress_buffer.append(stress_val)
                        stress_time_buffer.append(ts)
                        current_stress = stress_val

        # update latest_frame and bbox under lock
        with frame_lock:
            latest_frame = overlay_frame
            roi_bbox = current_roi

        frame_idx += 1

        # small sleep is not strictly required; camera.read throttles loop to ~FPS
        # but safe guard:
        time.sleep(0.001)

    # cleanup at thread end
    webcam.release()


# -----------------------
# Plotting / UI (matplotlib) - runs on main thread
# -----------------------
def start_live_plot():
    plt.ion()
    fig, (ax_signal, ax_stress) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # initialize empty lines
    sig_line, = ax_signal.plot([], [], linewidth=1.2)
    ax_signal.set_title("Raw rPPG Signal")
    ax_signal.set_xlabel("Time (s)")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.set_xlim(0, PLOT_WINDOW_SECONDS)
    ax_signal.set_ylim(-1.0, 1.0)
    ax_signal.grid(True)

    stress_line, = ax_stress.plot([], [], linewidth=1.5)
    ax_stress.set_title("Predicted Stress Index")
    ax_stress.set_xlabel("Time (s)")
    ax_stress.set_ylabel("Stress (0-100)")
    ax_stress.set_xlim(0, PLOT_WINDOW_SECONDS)
    ax_stress.set_ylim(0, 100)
    ax_stress.grid(True)

    # update function for FuncAnimation
    def update(_frame):
        # copy buffers under locks
        with rppg_lock:
            times = list(rppg_time_buffer)
            sigs = list(rppg_buffer)
        with stress_lock:
            st_times = list(stress_time_buffer)
            st_vals = list(stress_buffer)

        # A) update raw signal plot
        if len(times) > 0:
            t0 = times[-1] - PLOT_WINDOW_SECONDS
            xs = [t - t0 for t in times if t >= t0]
            ys = sigs[-len(xs):] if len(xs) > 0 else []
            # adjust axes
            ax_signal.set_xlim(0, min(PLOT_WINDOW_SECONDS, max(1.0, times[-1] - (times[0] if len(times) > 0 else 0))))
            # autoscale Y around recent data:
            if len(ys) > 0:
                y_min = min(ys)
                y_max = max(ys)
                margin = max(0.001, 0.1 * (y_max - y_min if y_max != y_min else 1.0))
                ax_signal.set_ylim(y_min - margin, y_max + margin)
                sig_line.set_data(xs, ys)

        # B) update stress plot (interpolate to same time base optionally)
        if len(st_times) > 0:
            t0s = st_times[-1] - PLOT_WINDOW_SECONDS
            xs_s = [t - t0s for t in st_times if t >= t0s]
            ys_s = st_vals[-len(xs_s):] if len(xs_s) > 0 else []
            ax_stress.set_xlim(0, min(PLOT_WINDOW_SECONDS, max(1.0, st_times[-1] - (st_times[0] if len(st_times) > 0 else 0))))
            if len(ys_s) > 0:
                stress_line.set_data(xs_s, ys_s)

        return sig_line, stress_line

    # Create animation object; disable frame cache to avoid memory growth
    anim = FuncAnimation(fig, update, interval=PLOT_UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)

    plt.show(block=False)
    return fig, anim


# -----------------------
# Helper: show video window (main thread) and handle quit
# -----------------------
def video_display_loop():
    global latest_frame
    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIDEO_WINDOW_NAME, VIDEO_WIDTH, VIDEO_HEIGHT)

    while running.is_set():
        with frame_lock:
            frame_to_show = latest_frame.copy() if latest_frame is not None else None

        if frame_to_show is not None:
            # show the frame
            cv2.imshow(VIDEO_WINDOW_NAME, frame_to_show)

        # key handling: press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running.clear()
            break

        time.sleep(0.01)

    cv2.destroyWindow(VIDEO_WINDOW_NAME)


# -----------------------
# Entrypoint
# -----------------------
def main():
    # start capture thread
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()

    # start plotting (matplotlib animation must run on main thread)
    fig, anim = start_live_plot()

    try:
        # concurrently show video window and keep matplotlib interactive
        video_display_loop()
    except KeyboardInterrupt:
        running.clear()
    finally:
        # ensure shutdown
        running.clear()
        cap_thread.join(timeout=1.0)
        plt.close(fig)


if __name__ == "__main__":
    main()
