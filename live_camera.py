import cv2
import collections
import torch
import numpy as np
import sys
import os
import threading
import tempfile
import uuid
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.inference import model, device, UCF50_CLASSES
from app.video_utils import get_inference_transform
from app.gradcam import gradcam_extractor, overlay_heatmap
from app.explanation import generate_explanation

# Global state for explainability threading
is_explaining = False
current_explanation = ""
explanation_timer = 0

def fetch_explanation(frames_list, heatmap, label):
    """Runs the Gemini explanation API in a background thread."""
    global is_explaining, current_explanation, explanation_timer
    
    try:
        temp_dir = tempfile.gettempdir()
        vid_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.mp4")
        hm_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
        
        # Write frames to temporary MP4
        height, width, _ = frames_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_path, fourcc, 10.0, (width, height))
        for f in frames_list:
            # Revert RGB back to BGR for OpenCV writing
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        
        # Write heatmap (ensure it's 0-255 uint8)
        heatmap_uint8 = np.uint8(255 * heatmap)
        cv2.imwrite(hm_path, heatmap_uint8)
        
        # API Call
        explanation = generate_explanation(vid_path, label, hm_path)
        current_explanation = explanation
        explanation_timer = 200  # Show text for roughly 200 frames
        
        # Cleanup
        if os.path.exists(vid_path): os.remove(vid_path)
        if os.path.exists(hm_path):  os.remove(hm_path)
            
    except Exception as e:
        print(f"Explainability Error: {e}")
    finally:
        is_explaining = False

def draw_wrapped_text(img, text, pos, font, font_scale, color, thickness, max_width):
    """Draws text on cv2 image with word wrapping based on pixel width."""
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        (w, h), _ = cv2.getTextSize(current_line + " " + word, font, font_scale, thickness)
        if w <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    x, y = pos
    (w, line_height), _ = cv2.getTextSize("H", font, font_scale, thickness)
    for line in lines:
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height + 10  # 10px line spacing

def main():
    global is_explaining, current_explanation, explanation_timer
    
    # Attempt to open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    transform = get_inference_transform()
    
    # Configuration for real-time performance
    SEQUENCE_LENGTH = 30
    FRAME_SKIP = 3  # Only add every 3rd frame to the buffer to cover more time
    INFERENCE_INTERVAL = 10  # Only run inference every 10 frames
    
    frames_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    
    # Temporal Smoothing: keeps track of the last N confident predictions
    prediction_history = collections.deque(maxlen=5)

    print("Starting webcam... Press 'q' to quit.")
    print("Press 'e' when observing a confident prediction to get a Mistral Explanation!")

    # Initialize a dummy heatmap for smooth overlays in case frames are buffering
    current_heatmap = None
    predicted_label = "Buffering..."
    display_label = "Buffering..."
    confidence = 0.0
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
            
        frame_count += 1

        # Convert BGR (OpenCV format) to RGB (for model preprocessing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only add every FRAME_SKIP-th frame to cover a wider temporal window
        if frame_count % FRAME_SKIP == 0:
            frames_buffer.append(frame_rgb)

        display_frame = frame.copy()

        # Only run inference if buffer is full and it's time (every INFERENCE_INTERVAL frames)
        if len(frames_buffer) == SEQUENCE_LENGTH and frame_count % INFERENCE_INTERVAL == 0:
            # 1. Motion Detection Heuristic
            # Filter completely static scenes (e.g. empty rooms) using OpenCV AbsDiff
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in list(frames_buffer)]
            diffs = [np.mean(cv2.absdiff(gray_frames[i], gray_frames[i-1])) for i in range(1, len(gray_frames))]
            avg_motion = np.mean(diffs) if diffs else 0
            
            # Increased threshold configuring how much motion is required to trigger model
            # 5.0 filters out standard background noise / lighting flickering
            MOTION_THRESHOLD = 5.0 
            
            if avg_motion < MOTION_THRESHOLD:
                predicted_label = "No Activity (Static Scene)"  # Shortened label
                confidence = 0.0
                current_heatmap = None
                prediction_history.clear() # Reset smoothing on static scenes
            else:
                try:
                    tensor_frames = [transform(Image.fromarray(f)) for f in frames_buffer]
                    sequence_tensor = torch.stack(tensor_frames).unsqueeze(0).to(device)

                    # Inference & Grad-CAM
                    heatmaps, target_class, conf, attention = gradcam_extractor.generate(sequence_tensor)
                    predicted_label = UCF50_CLASSES[target_class]
                    
                    # 2. Bias Mitigation
                    # The network treats "JugglingBalls" as a default class when viewing random slight arm movements.
                    # We penalize its raw confidence heavily to force it to be truly certain before picking it.
                    if predicted_label == "JugglingBalls":
                        conf -= 0.10
                    
                    # 3. Confidence Verification: Ensure the model is highly certain
                    CONFIDENCE_THRESHOLD = 0.65
                    if conf >= CONFIDENCE_THRESHOLD:
                        prediction_history.append(predicted_label)
                    else:
                        predicted_label = "Unrecognized"
                        prediction_history.append("Unrecognized")
                        
                    # 4. Temporal Smoothing (Majority Vote)
                    # We only display an action if it has been predicted mostly consistently 
                    # over the last few inference cycles, eliminating sudden 1-frame phantom flashes.
                    if len(prediction_history) > 0:
                        most_common_label = max(set(prediction_history), key=prediction_history.count)
                        
                        # Only lock in the label if it's the majority
                        if prediction_history.count(most_common_label) >= max(1, len(prediction_history) // 2):
                            display_label = most_common_label
                        else:
                            display_label = "Unrecognized (Transitioning)"
                    
                    current_heatmap = heatmaps[-1]
                    confidence = conf

                except Exception as e:
                    print(f"Error during inference: {e}")

        if current_heatmap is not None:
            # Overlay heatmap on the most recent frame for display
            display_frame = overlay_heatmap(frame_rgb, current_heatmap, alpha=0.5)
            text = f"{display_label} ({confidence:.2f})"
            color = (0, 255, 0) if confidence >= 0.65 and display_label not in ["Unrecognized", "No Activity (Static Scene)", "Unrecognized (Transitioning)"] else (0, 165, 255)
        else:
            if len(frames_buffer) < SEQUENCE_LENGTH:
                text = f"Buffering {len(frames_buffer)}/{SEQUENCE_LENGTH}"
                display_label = "Buffering..."
                color = (0, 0, 255)
            else:
                text = f"{predicted_label}"
                display_label = predicted_label
                color = (128, 128, 128)

        # Draw Prediction Background box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(display_frame, (8, 10), (12 + tw, 20 + th + 10), (0,0,0), -1)
        # Put prediction text on the frame
        cv2.putText(display_frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Determine if we should draw the Explanation
        if is_explaining:
            # Draw a waiting banner
            cv2.putText(display_frame, "Mistral is analyzing...", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif explanation_timer > 0 and current_explanation != "":
            # Draw the explanation text
            draw_wrapped_text(
                img=display_frame,
                text=current_explanation,
                pos=(10, display_frame.shape[0] - 120),  # near bottom
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.6,
                color=(255, 255, 255),
                thickness=1,
                max_width=display_frame.shape[1] - 20
            )
            explanation_timer -= 1

        # Display the result
        cv2.imshow("HAR Live Camera with Grad-CAM", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Trigger Explanation if confident and not already explaining
            if not is_explaining and confidence >= 0.70 and current_heatmap is not None:
                print("Triggering explanation...")
                is_explaining = True
                threading.Thread(
                    target=fetch_explanation, 
                    args=(list(frames_buffer), current_heatmap, predicted_label),
                    daemon=True
                ).start()
            elif is_explaining:
                print("Explanation already in progress...")
            else:
                print("Cannot explain: No confident activity detected yet.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

