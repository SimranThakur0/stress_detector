from src.video.webcam_stream import WebcamStream
from src.landmarks.face_mesh import FaceMeshDetector
from src.landmarks.eye_blink import BlinkDetector
from src.rppg.roi_extractor import extract_roi_frame, draw_roi
from src.rppg.rppg_extractor import RPPGExtractor
from src.fusion.stress_index import StressIndex
from src.visualization.overlay import draw_text
from src.visualization.graphs import LiveGraph

import cv2
import time


def main():
    # Initialize all modules
    webcam = WebcamStream()
    detector = FaceMeshDetector()
    blink_detector = BlinkDetector()

    rppg = RPPGExtractor(
        fs=30,
        window_size_seconds=12,
        region="forehead",
        use_pyvhr=False,
    )

    stress_model = StressIndex()
    graph = LiveGraph()

    last_graph_update = time.time()

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        results = detector.detect(frame)

        bpm = None
        blink_count = 0
        stress = 0.0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Blink detection (EAR)
            ear, blink_count = blink_detector.detect_blink(
                face.landmark, frame_w, frame_h
            )

            # Extract ROI (forehead recommended)
            roi_img, bbox = extract_roi_frame(
                frame, face.landmark, frame_w, frame_h, region="forehead"
            )

            if roi_img is not None:
                frame = draw_roi(frame, bbox, color=(0, 200, 0), thickness=2)

            # Add frame ROI to rPPG buffer
            rppg.push_frame(roi_img)

            # Estimate BPM when enough data is accumulated
            raw_sig, _ = rppg.get_raw_signal()
            if len(raw_sig) > 50:  # ~1.5 sec of signal at 30 FPS
                bpm = rppg.estimate_bpm()

            # Compute stress when BPM is available
            if bpm is not None:
                # blink_count is per-frame; your StressIndex will still
                # treat it as a weak secondary cue.
                stress = stress_model.compute(bpm, blink_count)

            # Update graphs every 0.3 seconds
            if time.time() - last_graph_update > 0.3:
                graph.add_values(bpm, stress)
                last_graph_update = time.time()

            # Draw UI text overlay
            frame = draw_text(frame, bpm, blink_count, stress)

            # EAR visual blink indicator
            if ear < blink_detector.ear_threshold:
                cv2.putText(
                    frame,
                    "BLINK",
                    (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        # Show window
        cv2.imshow("Real-Time Stress Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup webcam
    webcam.release()

    # After webcam closes → show the graph window
    graph.start()


if __name__ == "__main__":
    main()
