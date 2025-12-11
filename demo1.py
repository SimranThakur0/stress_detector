from src.video.webcam_stream import WebcamStream
from src.landmarks.face_mesh import FaceMeshDetector
from src.landmarks.eye_blink import BlinkDetector
from src.rppg.roi_extractor import extract_roi_frame, draw_roi
from src.rppg.rppg_extractor import RPPGExtractor
from src.fusion.stress_index import StressIndex
from src.visualization.overlay import draw_text

import cv2
import numpy as np
import time
import threading
import queue
from collections import deque


class RealtimeGraphRenderer:
    """Renders real-time graphs showing actual waveforms"""
    
    def __init__(self, width=1000, height=500):
        self.width = width
        self.height = height
        
        # Graph dimensions (split into 2 graphs)
        self.graph_height = height // 2
        self.margin = {'left': 70, 'right': 30, 'top': 50, 'bottom': 40}
        
        # Store complete raw signal buffer for waveform display
        self.raw_signal_buffer = deque(maxlen=360)  # 12 seconds at 30fps
        
        # Store time-series data for BPM and Stress
        self.max_history = 100  # Show last ~3 seconds of BPM/stress
        self.timestamps = deque(maxlen=self.max_history)
        self.bpm_history = deque(maxlen=self.max_history)
        self.stress_history = deque(maxlen=self.max_history)
        
        self.start_time = time.time()
    
    def add_raw_signal(self, raw_signal_array):
        """Add the entire raw signal array (replaces buffer each time)"""
        if raw_signal_array is not None and len(raw_signal_array) > 0:
            self.raw_signal_buffer = deque(raw_signal_array, maxlen=360)
    
    def add_metrics(self, bpm, stress):
        """Add BPM and stress values"""
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.bpm_history.append(bpm if bpm else 0)
        self.stress_history.append(stress if stress else 0)
    
    def render(self):
        """Render the complete visualization"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)  # Dark background
        
        # Draw Raw PPG Waveform (top)
        self._draw_waveform(canvas, 0)
        
        # Draw Stress Score over time (bottom)
        self._draw_stress_timeline(canvas, self.graph_height)
        
        return canvas
    
    def _draw_waveform(self, canvas, y_offset):
        """Draw the raw PPG signal waveform"""
        m = self.margin
        graph_w = self.width - m['left'] - m['right']
        graph_h = self.graph_height - m['top'] - m['bottom']
        
        # Background
        cv2.rectangle(canvas, 
                     (m['left'], y_offset + m['top']),
                     (self.width - m['right'], y_offset + self.graph_height - m['bottom']),
                     (35, 35, 35), -1)
        
        # Title
        cv2.putText(canvas, "Raw PPG Signal (rPPG Waveform)", 
                   (m['left'], y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Y-axis label
        cv2.putText(canvas, "Amplitude", (10, y_offset + self.graph_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if len(self.raw_signal_buffer) < 2:
            cv2.putText(canvas, "Collecting signal data...", 
                       (self.width//2 - 120, y_offset + self.graph_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            return
        
        # Convert to array and normalize
        signal = np.array(self.raw_signal_buffer)
        y_min, y_max = signal.min(), signal.max()
        
        if y_max - y_min < 0.01:  # Avoid division by zero
            return
        
        # Add padding
        y_range = y_max - y_min
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Draw grid
        for i in range(5):
            y = y_offset + m['top'] + int(i * graph_h / 4)
            cv2.line(canvas, (m['left'], y), (self.width - m['right'], y),
                    (50, 50, 50), 1)
            # Y-axis labels
            val = y_max - (y_max - y_min) * i / 4
            cv2.putText(canvas, f"{val:.1f}", (10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Convert signal to pixel coordinates
        points = []
        n_points = len(signal)
        for i, val in enumerate(signal):
            x = m['left'] + int((i / n_points) * graph_w)
            y_norm = (val - y_min) / (y_max - y_min)
            y = y_offset + m['top'] + graph_h - int(y_norm * graph_h)
            y = max(y_offset + m['top'], min(y_offset + self.graph_height - m['bottom'], y))
            points.append((x, y))
        
        # Draw the waveform
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (0, 255, 100), 2)
        
        # Show sample count
        cv2.putText(canvas, f"Samples: {len(signal)}", 
                   (self.width - m['right'] - 150, y_offset + self.graph_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def _draw_stress_timeline(self, canvas, y_offset):
        """Draw BPM and Stress over time"""
        m = self.margin
        graph_w = self.width - m['left'] - m['right']
        graph_h = self.graph_height - m['top'] - m['bottom']
        
        # Split into two sub-graphs (BPM top, Stress bottom)
        sub_h = graph_h // 2
        
        # --- BPM Graph ---
        bpm_y_offset = y_offset + m['top']
        
        # Background
        cv2.rectangle(canvas, 
                     (m['left'], bpm_y_offset),
                     (self.width - m['right'], bpm_y_offset + sub_h),
                     (35, 35, 35), -1)
        
        # Title
        cv2.putText(canvas, "Heart Rate (BPM)", 
                   (m['left'], y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        if len(self.bpm_history) > 1:
            # Draw BPM line
            bpm_points = self._create_timeline_points(
                list(self.bpm_history), 
                bpm_y_offset, 
                sub_h,
                y_min=40, 
                y_max=140
            )
            
            # Grid for BPM
            for val in [60, 80, 100, 120]:
                y = bpm_y_offset + sub_h - int(((val - 40) / 100) * sub_h)
                cv2.line(canvas, (m['left'], y), (self.width - m['right'], y),
                        (50, 50, 50), 1)
                cv2.putText(canvas, str(val), (m['left'] - 40, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            for i in range(len(bpm_points) - 1):
                cv2.line(canvas, bpm_points[i], bpm_points[i + 1], (100, 200, 255), 2)
            
            # Current value
            if self.bpm_history[-1] > 0:
                cv2.putText(canvas, f"{self.bpm_history[-1]:.0f} BPM", 
                           (self.width - m['right'] - 100, bpm_y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # --- Stress Graph ---
        stress_y_offset = bpm_y_offset + sub_h + 5
        
        # Background
        cv2.rectangle(canvas, 
                     (m['left'], stress_y_offset),
                     (self.width - m['right'], stress_y_offset + sub_h),
                     (35, 35, 35), -1)
        
        # Title
        cv2.putText(canvas, "Stress Index", 
                   (m['left'], stress_y_offset - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        if len(self.stress_history) > 1:
            # Draw stress line with fill
            stress_points = self._create_timeline_points(
                list(self.stress_history), 
                stress_y_offset, 
                sub_h,
                y_min=0, 
                y_max=100
            )
            
            # Grid for stress
            for val in [25, 50, 75]:
                y = stress_y_offset + sub_h - int((val / 100) * sub_h)
                cv2.line(canvas, (m['left'], y), (self.width - m['right'], y),
                        (50, 50, 50), 1)
                cv2.putText(canvas, str(val), (m['left'] - 40, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Fill under curve
            if len(stress_points) > 1:
                fill_points = stress_points.copy()
                fill_points.append((stress_points[-1][0], stress_y_offset + sub_h))
                fill_points.append((stress_points[0][0], stress_y_offset + sub_h))
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.array(fill_points, dtype=np.int32)], (255, 100, 100))
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # Draw line
            for i in range(len(stress_points) - 1):
                cv2.line(canvas, stress_points[i], stress_points[i + 1], (255, 100, 100), 2)
            
            # Current value
            if self.stress_history[-1] > 0:
                cv2.putText(canvas, f"{self.stress_history[-1]:.1f}", 
                           (self.width - m['right'] - 100, stress_y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # X-axis label
        cv2.putText(canvas, "Time (seconds)", 
                   (self.width//2 - 50, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _create_timeline_points(self, values, y_offset, height, y_min, y_max):
        """Convert time-series data to pixel coordinates"""
        m = self.margin
        graph_w = self.width - m['left'] - m['right']
        
        points = []
        n_points = len(values)
        for i, val in enumerate(values):
            x = m['left'] + int((i / n_points) * graph_w)
            y_norm = np.clip((val - y_min) / (y_max - y_min), 0, 1)
            y = y_offset + height - int(y_norm * height)
            points.append((x, y))
        
        return points


class ParallelStressDetection:
    def __init__(self):
        # Initialize modules
        self.webcam = WebcamStream()
        self.detector = FaceMeshDetector()
        self.blink_detector = BlinkDetector()
        
        self.rppg = RPPGExtractor(
            fs=30,
            window_size_seconds=12,
            region="forehead",
            use_pyvhr=False,
        )
        
        self.stress_model = StressIndex()
        self.graph_renderer = RealtimeGraphRenderer(width=1000, height=500)
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Control flags
        self.running = True
        self.last_graph_update = time.time()
        
        print("Real-Time Stress Detection Started")
        print("Press 'q' to quit")
        print("-" * 50)
    
    def processing_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            frame_h, frame_w = frame.shape[:2]
            results = self.detector.detect(frame)
            
            bpm = None
            blink_count = 0
            stress = 0.0
            processed_frame = frame.copy()
            
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                
                # Draw landmarks
                processed_frame = self.detector.draw_landmarks(processed_frame, results)
                
                # Blink detection
                ear, blink_count = self.blink_detector.detect_blink(
                    face.landmark, frame_w, frame_h
                )
                
                # Extract ROI
                roi_img, bbox = extract_roi_frame(
                    frame, face.landmark, frame_w, frame_h, region="forehead"
                )
                
                if roi_img is not None:
                    processed_frame = draw_roi(processed_frame, bbox, color=(0, 200, 0), thickness=2)
                    
                    # Add frame to rPPG buffer
                    self.rppg.push_frame(roi_img)
                    
                    # Get raw signal for waveform display
                    raw_sig, _ = self.rppg.get_raw_signal()
                    
                    # Update waveform display with entire signal buffer
                    if len(raw_sig) > 10:
                        self.graph_renderer.add_raw_signal(raw_sig)
                    
                    # Estimate BPM when enough data
                    if len(raw_sig) > 50:
                        bpm = self.rppg.estimate_bpm()
                    
                    # Compute stress
                    if bpm is not None:
                        stress = self.stress_model.compute(bpm, blink_count)
                    
                    # Update metrics timeline
                    current_time = time.time()
                    if current_time - self.last_graph_update > 0.1:
                        self.graph_renderer.add_metrics(bpm, stress)
                        self.last_graph_update = current_time
                    
                    # Draw text overlay
                    processed_frame = draw_text(processed_frame, bpm, blink_count, stress)
                    
                    # Blink indicator
                    if ear < self.blink_detector.ear_threshold:
                        cv2.putText(processed_frame, "BLINK", (200, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Put result in queue
            try:
                self.result_queue.put(processed_frame, block=False)
            except queue.Full:
                pass
    
    def run(self):
        """Main run loop"""
        # Start processing thread
        proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        proc_thread.start()
        
        while self.running:
            # Read frame
            ret, frame = self.webcam.read()
            if not ret:
                break
            
            # Queue for processing
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
            
            # Get processed frame
            try:
                processed_frame = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                processed_frame = frame
            
            # Render graphs
            graph_image = self.graph_renderer.render()
            
            # Create combined view
            display_width = 1000
            
            # Resize webcam
            aspect_ratio = processed_frame.shape[0] / processed_frame.shape[1]
            webcam_height = int(display_width * aspect_ratio)
            webcam_display = cv2.resize(processed_frame, (display_width, webcam_height))
            
            # Combine vertically: webcam on top, graphs below
            combined = np.vstack([webcam_display, graph_image])
            
            # Display
            cv2.imshow("Real-Time Stress Detection - Webcam + Graphs", combined)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nShutting down...")
                self.running = False
                break
        
        # Cleanup
        self.running = False
        proc_thread.join(timeout=2.0)
        self.webcam.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")


def main():
    detector = ParallelStressDetection()
    detector.run()


if __name__ == "__main__":
    main()