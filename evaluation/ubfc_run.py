# evaluation/ubfc_run.py
import cv2
import numpy as np
from src.landmarks.face_mesh import FaceMeshDetector
from src.rppg.roi_extractor import extract_roi_frame
from src.rppg.rppg_extractor import RPPGExtractor

def run_on_ubfc(video_path):
    cap = cv2.VideoCapture(video_path)
    mesh = FaceMeshDetector()
    rppg = RPPGExtractor(fs=30, window_size_seconds=12)

    bpm_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        results = mesh.detect(frame)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            roi, _ = extract_roi_frame(frame, face.landmark, frame_w, frame_h)
            rppg.push_frame(roi)

            raw_sig, _ = rppg.get_raw_signal()
            if len(raw_sig) > 50:
                bpm = rppg.estimate_bpm()
                if bpm:
                    bpm_values.append(bpm)

    cap.release()
    return np.array(bpm_values)


if __name__ == "__main__":
    video_file = "D:\\stress_detector\\eval_data\\subject1\\vid.avi"
    bpm_results = run_on_ubfc(video_file)
    print("Estimated BPM values:", bpm_results)
