# src/pipeline.py
import cv2
import time

from video.webcam_stream import WebcamStream
from landmarks.face_mesh import FaceMeshDetector
from landmarks.eye_blink import BlinkDetector
from rppg.roi_extractor import extract_roi_frame, draw_roi
from rppg.rppg_extractor import RPPGExtractor
from fusion.stress_index import StressIndex
from visualization.overlay import draw_text
from visualization.graphs import LiveGraph

class StressPipeline:
    def __init__(self):
        self.webcam = WebcamStream()
        self.mesh = FaceMeshDetector()
        self.blink = BlinkDetector()
        self.rppg = RPPGExtractor(fs=30, window_size_seconds=12)
        self.stress_model = StressIndex()

        self.graph = LiveGraph()
        self.last_graph_update = time.time()

    def run(self):
        while True:
            ret, frame = self.webcam.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]

            results = self.mesh.detect(frame)

            bpm = None
            stress = None
            blink_count = 0

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                frame = self.mesh.draw_landmarks(frame, results)

                # Blink detection
                ear, blink_count = self.blink.detect_blink(
                    face.landmark, frame_w, frame_h
                )

                # ROI extraction
                roi, bbox = extract_roi_frame(
                    frame, face.landmark, frame_w, frame_h, region="forehead"
                )
                frame = draw_roi(frame, bbox)

                # Push ROI to rPPG
                self.rppg.push_frame(roi)

                raw_sig, _ = self.rppg.get_raw_signal()
                if len(raw_sig) > 50:  # at least ~1.5 sec
                    bpm = self.rppg.estimate_bpm()

                # Stress calculation
                if bpm is not None:
                    stress = self.stress_model.compute(bpm, blink_count)

                # Update graphs every 0.3s
                if time.time() - self.last_graph_update > 0.3:
                    self.graph.add_values(bpm, stress)
                    self.last_graph_update = time.time()

            # Draw UI text
            frame = draw_text(frame, bpm, blink_count, stress if stress else 0)

            cv2.imshow("Stress Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        self.graph.start()
