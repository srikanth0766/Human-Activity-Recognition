import torch
import sys
import os

# Ensure imports work regardless of execution location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_loader import load_model
from app.video_utils import preprocess_video

# UCF50 Classes mapped exactly as they were in the training stage
UCF50_CLASSES = [
    "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards",
    "BreastStroke", "CleanAndJerk", "Diving", "Drumming", "Fencing",
    "GolfSwing", "HighJump", "HorseRace", "HorseRiding", "HulaHoop",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking",
    "Lunges", "MilitaryParade", "Mixing", "Nunchucks", "PizzaTossing",
    "PlayingGuitar", "PlayingPiano", "PlayingTabla", "PlayingViolin", "PoleVault",
    "PommelHorse", "PullUps", "Punch", "PushUps", "RockClimbingIndoor",
    "RopeClimbing", "Rowing", "SalsaSpin", "SkateBoarding", "Skiing",
    "Skijet", "SoccerJuggling", "Swing", "TaiChi", "TennisSwing",
    "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog", "YoYo"
]

# Initialize model once at module load
print("Initializing HAR model for inference...")
model, device = load_model()

def predict(video_path, num_frames=30):
    """
    Run inference on a video path.
    1. Load video
    2. Preprocess frames
    3. Forward pass through model
    4. Apply softmax
    5. Get predicted class index
    6. Return predicted label & confidence score
    """
    try:
        # Preprocess video into [1, T, 3, 224, 224] tensor
        input_tensor = preprocess_video(video_path, num_frames)
        input_tensor = input_tensor.to(device)

        # Forward pass without gradients
        with torch.no_grad():
            logits = model(input_tensor)
            
            # Apply softmax
            probs = torch.softmax(logits, dim=1)
            
            # Get max prediction
            confidence, pred_idx = probs.max(dim=1)
            
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            
            # Map index to class label
            predicted_label = UCF50_CLASSES[pred_idx]
            
            return predicted_label, confidence
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
