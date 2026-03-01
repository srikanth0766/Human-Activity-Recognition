import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def get_inference_transform():
    """Returns the standard torchvision transform matching ImageNet norms"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_and_sample_frames(video_path, num_frames=30):
    """
    Load video using OpenCV, extract uniform frames, transform to tensors.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video has fewer frames than requested, 
    # we just take all, otherwise we sample uniformly
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Fallback if CAP_PROP_FRAME_COUNT fails
        indices = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) > 500: # safety limit
                break
        total_frames = len(frames)
        if total_frames == 0:
            return []
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    return frames

def preprocess_video(video_path, num_frames=30):
    """
    Takes a video path, extracts frames, and applies transforms.
    Returns tensor of shape [1, T, 3, 224, 224]
    """
    frames = extract_and_sample_frames(video_path, num_frames)
    if not frames:
        raise ValueError("No frames extracted from video.")

    # Convert to PIL Image for torchvision transforms
    transform = get_inference_transform()
    
    tensor_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        tensor_img = transform(pil_img)
        tensor_frames.append(tensor_img)

    # Stack along temporal dimension: [T, C, H, W]
    sequence_tensor = torch.stack(tensor_frames)
    
    # Add batch dimension: [1, T, C, H, W]
    sequence_tensor = sequence_tensor.unsqueeze(0)
    
    return sequence_tensor
