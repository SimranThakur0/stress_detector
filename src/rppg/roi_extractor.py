# src/rppg/roi_extractor.py
import numpy as np
import cv2

# Default landmark indices for forehead/cheek region (MediaPipe 468 pts)
# We'll compute bounding box from a few stable landmarks: forehead / cheeks / nose root
ROI_LANDMARKS = {
    "left_cheek": [234, 93, 132],   # approximate
    "right_cheek": [454, 323, 361], # approximate
    "forehead": [10, 338, 127]      # approximate center/top
}

def landmarks_to_bbox(landmarks, indices, frame_w, frame_h, pad=0.25):
    """Return bbox (x, y, w, h) from given landmark indices (normalized coords)."""
    pts = []
    for i in indices:
        lm = landmarks[i]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        pts.append((x, y))
    pts = np.array(pts)
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    y_max = int(np.max(pts[:, 1]))
    w = x_max - x_min
    h = y_max - y_min

    # pad
    x_pad = int(w * pad)
    y_pad = int(h * pad)
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(frame_w - 1, x_max + x_pad)
    y2 = min(frame_h - 1, y_max + y_pad)
    return x1, y1, x2 - x1, y2 - y1

def extract_roi_frame(frame, face_landmarks, frame_w, frame_h, region="forehead"):
    """
    Returns ROI image (numpy array) for the region (forehead/left_cheek/right_cheek).
    face_landmarks: mediapipe normalized landmarks (list-like)
    """
    if region not in ROI_LANDMARKS:
        region = "forehead"
    bbox = landmarks_to_bbox(face_landmarks, ROI_LANDMARKS[region], frame_w, frame_h, pad=0.35)
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None, bbox
    roi = frame[y:y+h, x:x+w]
    return roi, bbox


def draw_roi(frame, bbox, color=(0,255,0), thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    return frame

