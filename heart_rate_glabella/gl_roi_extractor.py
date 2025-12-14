# src/rppg/roi_extractor.py
import numpy as np
import cv2

# Correct landmark indices based on research paper:
# "Optimal facial regions for remote heart rate measurement during physical and cognitive activities"
# Nature npj Cardiovascular Health (2024)
# https://www.nature.com/articles/s44325-024-00033-7

ROI_LANDMARKS = {
    # FOREHEAD REGIONS (most commonly used for rPPG)
    "glabella": [9, 107, 66, 105, 63, 70, 156, 143, 116, 123, 147, 213, 
                 192, 214, 210, 211, 32, 208, 251, 284, 332, 297, 338, 
                 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 389, 
                 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 
                 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 
                 93, 234, 127, 162],
    
    "medial_forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 
                        323, 361, 288, 397, 365, 379, 378, 400, 377, 
                        152, 148, 176, 149, 150, 136, 172, 58, 132, 
                        93, 234, 127, 162, 21, 54, 103, 67, 109],
    
    "left_lateral_forehead": [108, 69, 104, 68, 71, 139, 34, 227, 
                              137, 177, 215, 138, 135, 169, 170, 
                              140, 171, 175, 396],
    
    "right_lateral_forehead": [337, 299, 333, 298, 301, 368, 264, 
                               447, 366, 401, 435, 367, 364, 394, 
                               395, 369, 396, 400, 175],
    
    # CHEEK REGIONS
    "left_cheek": [116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 
                   199, 428, 204, 36, 142, 126, 217, 47, 114],
    
    "right_cheek": [345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 
                    419, 204, 424, 266, 371, 355, 437, 277, 343],
    
    # ORIGINAL APPROXIMATE REGIONS (for backward compatibility)
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
                 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
                 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 
                 162, 21, 54, 103, 67, 109],
}


def landmarks_to_bbox(landmarks, indices, frame_w, frame_h, pad=0.2):
    """
    Return bbox (x, y, w, h) from given landmark indices (normalized coords).
    
    Args:
        landmarks: MediaPipe face landmarks (normalized 0-1)
        indices: List of landmark indices to use for ROI
        frame_w: Frame width in pixels
        frame_h: Frame height in pixels
        pad: Padding factor (default 0.2 for 20% padding)
    
    Returns:
        tuple: (x, y, w, h) bounding box coordinates
    """
    pts = []
    for i in indices:
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        pts.append((x, y))
    
    if len(pts) == 0:
        return 0, 0, 0, 0
    
    pts = np.array(pts)
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    y_max = int(np.max(pts[:, 1]))
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Apply padding
    x_pad = int(w * pad)
    y_pad = int(h * pad)
    
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(frame_w - 1, x_max + x_pad)
    y2 = min(frame_h - 1, y_max + y_pad)
    
    return x1, y1, x2 - x1, y2 - y1


def extract_roi_frame(frame, face_landmarks, frame_w, frame_h, region="forehead"):
    """
    Extract ROI image for the specified region (forehead/glabella/cheeks).
    
    Research shows that glabella and medial_forehead regions provide the best
    results for rPPG heart rate detection across different motion types.
    
    Args:
        frame: Input video frame (numpy array)
        face_landmarks: MediaPipe normalized landmarks (list-like)
        frame_w: Frame width
        frame_h: Frame height
        region: ROI region name (default "forehead")
                Options: "glabella", "medial_forehead", "left_lateral_forehead",
                        "right_lateral_forehead", "left_cheek", "right_cheek", 
                        "forehead" (alias for medial_forehead)
    
    Returns:
        tuple: (roi_image, bbox) where roi_image is the extracted region
               and bbox is (x, y, w, h)
    """
    if region not in ROI_LANDMARKS:
        region = "forehead"
    
    # Use appropriate padding based on region
    # Forehead regions need less padding, cheeks need more
    pad = 0.15 if "forehead" in region or region == "glabella" else 0.25
    
    bbox = landmarks_to_bbox(face_landmarks, ROI_LANDMARKS[region], 
                            frame_w, frame_h, pad=pad)
    x, y, w, h = bbox
    
    if w <= 0 or h <= 0:
        return None, bbox
    
    roi = frame[y:y+h, x:x+w]
    return roi, bbox


def draw_roi(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw ROI bounding box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box (x, y, w, h)
        color: Box color (default green)
        thickness: Line thickness (default 2)
    
    Returns:
        Frame with drawn bounding box
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    return frame


def get_available_regions():
    """
    Get list of available ROI region names.
    
    Returns:
        list: Available region names
    """
    return list(ROI_LANDMARKS.keys())


def get_recommended_region():
    """
    Get the recommended region for rPPG based on latest research.
    
    According to research (Nature 2024), the glabella region provides
    the best overall performance for heart rate detection across different
    motion types and cognitive tasks.
    
    Returns:
        str: Recommended region name
    """
    return "glabella"
