from src.video.webcam_stream import WebcamStream
from src.landmarks.face_mesh import FaceMeshDetector
from src.landmarks.eye_blink import BlinkDetector
from heart_rate_glabella.gl_roi_extractor import extract_roi_frame, draw_roi, get_recommended_region
from HRV.hrv_rppg_extractor import RPPGExtractor
from HRV.hrv_analyzer import HRVAnalyzer
from HRV.hrv_stress_index import StressIndex
from src.visualization.overlay import draw_text
from src.visualization.graphs import LiveGraph

import cv2
import time
import numpy as np


def filter_rr_intervals_aggressive(rr_intervals):
    """
    AGGRESSIVE filtering of RR intervals to remove artifacts.
    This is the KEY function to fix your high HRV values.
    """
    if len(rr_intervals) < 3:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    
    print(f"  [Filter] Input: {len(rr_array)} intervals, range: {rr_array.min():.0f}-{rr_array.max():.0f} ms")
    
    # Step 1: Remove physiologically impossible values
    # Stricter bounds: 400-1500ms (40-150 BPM range)
    valid_mask = (rr_array >= 400) & (rr_array <= 1500)
    filtered = rr_array[valid_mask]
    
    print(f"  [Filter] After range filter: {len(filtered)} intervals")
    
    if len(filtered) < 3:
        return filtered.tolist()
    
    # Step 2: Remove values that change too rapidly
    # Maximum 20% change between consecutive intervals
    stable_filtered = [filtered[0]]
    for i in range(1, len(filtered)):
        change_ratio = abs(filtered[i] - filtered[i-1]) / filtered[i-1]
        if change_ratio < 0.20:  # 20% max change
            stable_filtered.append(filtered[i])
    
    filtered = np.array(stable_filtered)
    
    print(f"  [Filter] After stability filter: {len(filtered)} intervals")
    
    if len(filtered) < 3:
        return filtered.tolist()
    
    # Step 3: Statistical outlier removal using STRICTER IQR method
    q1 = np.percentile(filtered, 25)
    q3 = np.percentile(filtered, 75)
    iqr = q3 - q1
    
    # Use 1.0 * IQR instead of 1.5 * IQR (more aggressive)
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    
    filtered = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)]
    
    print(f"  [Filter] After IQR filter: {len(filtered)} intervals")
    
    # Step 4: Remove values far from median
    if len(filtered) >= 5:
        median = np.median(filtered)
        mad = np.median(np.abs(filtered - median))
        if mad > 0:
            # Keep only values within 2.5 MAD of median
            filtered = filtered[np.abs(filtered - median) <= 2.5 * mad]
    
    print(f"  [Filter] Final: {len(filtered)} intervals, mean: {np.mean(filtered):.0f} ms")
    
    return filtered.tolist()


def draw_hrv_overlay(frame, hrv_metrics, hrv_stress, x=10, y=120):
    """Draw HRV information overlay on frame."""
    if hrv_metrics is None:
        cv2.putText(frame, "HRV: Collecting data...", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return frame
    
    time_domain = hrv_metrics.get('time_domain', {})
    quality = hrv_metrics.get('data_quality', {})
    
    # Draw HRV stress score
    stress_color = (0, 255, 0) if hrv_stress < 40 else (0, 165, 255) if hrv_stress < 70 else (0, 0, 255)
    cv2.putText(frame, f"HRV Stress: {hrv_stress:.1f}", (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_color, 2)
    
    # Draw SDNN (key HRV metric)
    sdnn = time_domain.get('sdnn', 0)
    sdnn_color = (0, 255, 0) if sdnn > 50 else (0, 165, 255) if sdnn > 30 else (0, 0, 255)
    cv2.putText(frame, f"SDNN: {sdnn:.1f}ms", (x, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, sdnn_color, 1)
    
    # Draw RMSSD
    rmssd = time_domain.get('rmssd', 0)
    rmssd_color = (0, 255, 0) if rmssd > 40 else (0, 165, 255) if rmssd > 20 else (0, 0, 255)
    cv2.putText(frame, f"RMSSD: {rmssd:.1f}ms", (x, y + 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, rmssd_color, 1)
    
    # Draw data quality indicator
    num_intervals = quality.get('num_rr_intervals', 0)
    quality_text = f"RR intervals: {num_intervals}"
    quality_color = (0, 255, 0) if num_intervals >= 30 else (0, 165, 255) if num_intervals >= 10 else (128, 128, 128)
    cv2.putText(frame, quality_text, (x, y + 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)
    
    return frame


def main():
    # Initialize all modules
    webcam = WebcamStream()
    detector = FaceMeshDetector()
    blink_detector = BlinkDetector()

    # Use the recommended region (glabella) for best results
    recommended_region = get_recommended_region()
    
    print("="*60)
    print("🫀 REAL-TIME STRESS DETECTION WITH HRV (FIXED)")
    print("="*60)
    print(f"📍 Using ROI region: {recommended_region}")
    print("💡 Glabella (between eyebrows) provides best accuracy")
    print("\n📊 Features:")
    print("   • Heart Rate (BPM)")
    print("   • Heart Rate Variability (HRV)")
    print("   • Aggressive RR interval filtering")
    print("   • Stress Index (HR + HRV + Blink)")
    print("\n⚠️  For best HRV accuracy:")
    print("   • Stay still for 30-60 seconds")
    print("   • Good lighting (>500 lux)")
    print("   • Face camera directly")
    print("\n🎯 Press 'q' to quit, 'r' to reset HRV\n")
    print("="*60 + "\n")

    rppg = RPPGExtractor(
        fs=30,
        window_size_seconds=12,
        region=recommended_region,
        use_pyvhr=False,
    )

    # Initialize stress and HRV analyzers
    stress_model = StressIndex()
    hrv_analyzer = HRVAnalyzer(min_rr_count=10)
    
    graph = LiveGraph()

    # Timing variables
    last_graph_update = time.time()
    last_hrv_update = time.time()
    last_hrv_print = time.time()
    
    # HRV data storage
    hrv_metrics = None
    hrv_stress = None
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        results = detector.detect(frame)

        bpm = None
        blink_count = 0
        stress = 0.0
        frame_count += 1

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Blink detection (EAR)
            ear, blink_count = blink_detector.detect_blink(
                face.landmark, frame_w, frame_h
            )

            # Extract ROI using glabella region
            roi_img, bbox = extract_roi_frame(
                frame, face.landmark, frame_w, frame_h, 
                region=recommended_region
            )

            if roi_img is not None:
                # Draw ROI with orange color
                frame = draw_roi(frame, bbox, color=(255, 165, 0), thickness=2)

            # Add frame ROI to rPPG buffer
            rppg.push_frame(roi_img)

            # Estimate BPM when enough data is accumulated
            raw_sig, _ = rppg.get_raw_signal()
            if len(raw_sig) > 50:  # ~1.5 sec of signal at 30 FPS
                bpm = rppg.estimate_bpm()

            # === HRV ANALYSIS (every 5 seconds) ===
            if time.time() - last_hrv_update > 5.0:
                # Get HRV data from rPPG
                hrv_data = rppg.get_hrv_data(update=True)
                
                if hrv_data['data_sufficient']:
                    # *** THIS IS THE KEY FIX ***
                    # Apply aggressive filtering to RR intervals
                    raw_rr = hrv_data['rr_intervals']
                    filtered_rr = filter_rr_intervals_aggressive(raw_rr)
                    
                    # Only proceed if we have enough filtered intervals
                    if len(filtered_rr) >= 10:
                        # Calculate HRV metrics with FILTERED intervals
                        hrv_metrics = hrv_analyzer.calculate_all_metrics(filtered_rr)
                        
                        if hrv_metrics:
                            # Get HRV-based stress score
                            hrv_stress = hrv_analyzer.get_stress_from_hrv(hrv_metrics)
                            
                            # Check for still-abnormal values
                            td = hrv_metrics['time_domain']
                            sdnn = td.get('sdnn', 0)
                            rmssd = td.get('rmssd', 0)
                            
                            if sdnn > 100 or rmssd > 100:
                                print(f"⚠️  Warning: High HRV values (SDNN={sdnn:.1f}, RMSSD={rmssd:.1f})")
                                print("   Consider: better lighting, staying more still\n")
                            
                            # Print detailed HRV report every 15 seconds
                            if time.time() - last_hrv_print > 15.0:
                                hrv_analyzer.print_hrv_report(hrv_metrics)
                                last_hrv_print = time.time()
                    else:
                        print(f"📊 Not enough quality RR intervals ({len(filtered_rr)}/10 after filtering)")
                else:
                    # Show progress
                    print(f"📊 Collecting HRV data... ({hrv_data['num_intervals']}/10 RR intervals)")
                
                last_hrv_update = time.time()

            # === STRESS COMPUTATION ===
            if bpm is not None:
                # Primary stress index (HR + HR trend + blink)
                stress = stress_model.compute(bpm, blink_count)
                
                # If we have HRV, we can combine both stress scores
                if hrv_stress is not None:
                    # Weighted combination: 60% HRV-based, 40% HR-based
                    combined_stress = 0.6 * hrv_stress + 0.4 * stress
                    stress = combined_stress

            # Update graphs every 0.3 seconds
            if time.time() - last_graph_update > 0.3:
                graph.add_values(bpm, stress)
                last_graph_update = time.time()

            # === DRAW UI OVERLAYS ===
            
            # Main metrics (BPM, Blink, Stress)
            frame = draw_text(frame, bpm, blink_count, stress)

            # HRV overlay
            if hrv_metrics is not None and hrv_stress is not None:
                frame = draw_hrv_overlay(frame, hrv_metrics, hrv_stress)

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

            # ROI region label
            cv2.putText(
                frame,
                f"ROI: {recommended_region.upper()}",
                (10, frame_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 165, 0),
                1,
            )
            
            # Signal quality indicator
            raw_sig, _ = rppg.get_raw_signal()
            if len(raw_sig) >= 30:
                try:
                    signal_quality = rppg.get_signal_quality()
                    quality_color = (0, 255, 0) if signal_quality['score'] > 0.7 else \
                                   (0, 165, 255) if signal_quality['score'] > 0.4 else (0, 0, 255)
                    cv2.putText(
                        frame,
                        f"Signal: {signal_quality['quality']}",
                        (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        quality_color,
                        1,
                    )
                except ValueError:
                    cv2.putText(
                        frame,
                        "Signal: Initializing...",
                        (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 128),
                        1,
                    )
            else:
                cv2.putText(
                    frame,
                    "Signal: Initializing...",
                    (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    1,
                )
            
            # FPS counter
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (frame_w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Show window
        cv2.imshow("Real-Time Stress Detection with HRV", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset HRV data
            print("\n🔄 Resetting HRV data...\n")
            rppg.reset()
            stress_model.reset()
            hrv_metrics = None
            hrv_stress = None
            last_hrv_update = time.time()

    # Cleanup
    webcam.release()

    # Print final summary
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Total duration: {elapsed:.1f} seconds")
    
    if hrv_metrics:
        print("\n📈 Final HRV Metrics:")
        td = hrv_metrics['time_domain']
        print(f"   Mean HR: {td['mean_hr']:.1f} BPM")
        print(f"   SDNN: {td['sdnn']:.1f} ms")
        print(f"   RMSSD: {td['rmssd']:.1f} ms")
        if hrv_stress:
            print(f"   HRV Stress: {hrv_stress:.1f}/100")
        
        # Quality assessment
        if td['sdnn'] < 30:
            print("\n⚠️  Note: SDNN is low - this may indicate:")
            print("   - High stress (legitimate)")
            print("   - Poor signal quality")
            print("   - Insufficient data collection time")
        elif td['sdnn'] > 100:
            print("\n⚠️  Note: SDNN is still high - recommendations:")
            print("   - Improve lighting significantly")
            print("   - Stay completely motionless")
            print("   - Increase data collection time to 60+ seconds")
    
    print("="*60 + "\n")

    # Show the graph window
    graph.start()


if __name__ == "__main__":
    main()