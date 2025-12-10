import cv2

class WebcamStream:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
