import torch
import os
import sys

# Add models path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture import HARModel


def load_model():
    """
    Initializes the HAR Model architecture and loads best_model.pth.
    Returns:
        model: HARModel in eval mode.
        device: torch.device used.
    """
    # 1. Initialize architecture
    # pretrained_backbone=False per mega prompt since we load our own weights
    model = HARModel(num_classes=50, pretrained_backbone=False)

    # 2. Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. Load weights/best_model.pth
    # Construct absolute path ensuring safety from where it's executed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "weights", "best_model.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    print(f"Loading HAR Model weights from {weights_path} onto {device}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Note: If the checkpoint was saved as a whole dict vs just state_dict, handle it
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback if just state dict
        model.load_state_dict(checkpoint)

    # 4. Set to eval and move to device
    model = model.to(device)
    model.eval()

    print("Model successfully loaded and set to evaluation mode.")
    return model, device
