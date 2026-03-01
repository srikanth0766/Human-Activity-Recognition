import os
import sys
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure imports work regardless of execution location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import model, device, UCF50_CLASSES
from app.video_utils import preprocess_video, extract_and_sample_frames

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "outputs")

class GradCAM:
    """
    Grad-CAM targeting ResNet-50 layer4.
    """
    def __init__(self, model):
        self.model = model
        self.target_layer = model.backbone.layer4
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmaps for a video input [1, T, C, H, W]
        """
        input_tensor = input_tensor.requires_grad_(True)
        self.model.eval()

        # Forward
        logits, attn_weights = self.model(input_tensor, return_attention=True)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()

        # Backward
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        seq_len = input_tensor.shape[1]
        heatmaps = []

        for t in range(seq_len):
            if self.activations is not None and self.gradients is not None:
                act = self.activations[t]
                grad = self.gradients[t]
                weights = grad.mean(dim=[1, 2])

                cam = torch.zeros(act.shape[1:], device=device)
                for i, w in enumerate(weights):
                    cam += w * act[i]

                cam = F.relu(cam).cpu().numpy()
                if cam.max() > 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                heatmaps.append(cam)
            else:
                heatmaps.append(np.zeros((7, 7)))

        # attention array shape [T]
        attention = attn_weights[0].detach().cpu().numpy()
        return heatmaps, target_class, confidence, attention


def overlay_heatmap(frame_rgb, heatmap, alpha=0.4):
    """
    Overlay heatmap onto original frame.
    """
    h, w = frame_rgb.shape[:2]
    
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Convert original RGB back to BGR for cv2 operations
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


# Global GradCAM instance
gradcam_extractor = GradCAM(model)

def predict_with_gradcam(video_path, num_frames=30):
    """
    1. Preprocess frames
    2. Extract heatmap
    3. Save the most-attended frame overlay
    4. Return predicted class, confidence, saved path
    """
    frames = extract_and_sample_frames(video_path, num_frames=num_frames)
    if not frames:
        raise ValueError("Could not extract frames for Grad-CAM.")
        
    input_tensor = preprocess_video(video_path, num_frames=num_frames).to(device)
    
    # Generate Heatmaps
    heatmaps, target_class, confidence, attention = gradcam_extractor.generate(input_tensor)
    
    predicted_label = UCF50_CLASSES[target_class]
    
    # Let's save the frame with the highest attention score
    best_frame_idx = np.argmax(attention)
    best_frame_rgb = frames[best_frame_idx]
    best_heatmap = heatmaps[best_frame_idx]
    
    overlaid_frame = overlay_heatmap(best_frame_rgb, best_heatmap)
    
    # Save the file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_filename = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    cv2.imwrite(out_path, overlaid_frame)
    
    return predicted_label, confidence, out_path

