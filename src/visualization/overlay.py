# src/visualization/overlay.py
import cv2

def draw_text(frame, bpm, blink_count, stress):
    cv2.putText(frame, f"BPM: {bpm if bpm else '--'}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Stress: {stress:.1f}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame
