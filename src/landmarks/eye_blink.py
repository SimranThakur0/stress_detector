import numpy as np

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        self.frame_counter = 0
        self.total_blinks = 0

        # Mediapipe eye landmark indices
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def compute_ear(self, eye_landmarks):
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        vertical = (self.euclidean_distance(p2, p6) + self.euclidean_distance(p3, p5))
        horizontal = self.euclidean_distance(p1, p4)
        ear = vertical / (2.0 * horizontal)
        return ear

    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        eye_points = []
        for idx in indices:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            eye_points.append(np.array([x, y]))
        return eye_points

    def detect_blink(self, face_landmarks, frame_w, frame_h):
        # extract left and right eye landmarks
        left_eye = self.get_eye_landmarks(face_landmarks, self.left_eye_idx, frame_w, frame_h)
        right_eye = self.get_eye_landmarks(face_landmarks, self.right_eye_idx, frame_w, frame_h)

        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Blink logic
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consecutive_frames:
                self.total_blinks += 1
            self.frame_counter = 0

        return ear, self.total_blinks
