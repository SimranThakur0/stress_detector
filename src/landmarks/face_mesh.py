import mediapipe as mp
import cv2

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return results

    def draw_landmarks(self, frame, results):
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                for lm in face.landmark:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
