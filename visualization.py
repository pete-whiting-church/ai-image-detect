"""
Attention visualization using Grad-CAM for explainability.
Shows which regions of the image influenced the AI detection decision.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class AttentionVisualizer:
    """
    Generates attention heatmaps showing which image regions
    are most indicative of AI generation.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "model.visual.transformer.resblocks[-1]"):
        """
        Initialize attention visualizer.

        Args:
            model: The detector model
            target_layer_name: Name of layer to visualize (for CLIP, use the last transformer block)
        """
        self.model = model

        # For CLIP models, target the last transformer residual block
        try:
            # Access the CLIP vision transformer's last block
            target_layers = [model.model.visual.transformer.resblocks[-1].ln_1]
        except AttributeError:
            # Fallback if structure is different
            print("Warning: Could not find CLIP transformer layers, using model output")
            target_layers = None

        if target_layers:
            self.cam = GradCAM(
                model=model,
                target_layers=target_layers,
                reshape_transform=self._reshape_transform
            )
        else:
            self.cam = None

    def _reshape_transform(self, tensor, height=14, width=14):
        """
        Reshape CLIP ViT output for CAM visualization.

        CLIP ViT-L/14 outputs [batch, num_patches+1, hidden_dim]
        We need to reshape to [batch, hidden_dim, height, width]
        """
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, width,
                                          tensor.size(2))

        # [batch, height, width, channels] -> [batch, channels, height, width]
        result = result.transpose(2, 3).transpose(1, 2)

        return result

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Generate attention heatmap for an image.

        Args:
            image_tensor: Preprocessed image tensor [1, 3, H, W]
            original_image: Original image as numpy array [H, W, 3] in [0,1] range

        Returns:
            Heatmap visualization overlaid on original image, or None if unavailable
        """
        if self.cam is None:
            return None

        # Ensure image is in [0,1] range
        if original_image.max() > 1.0:
            original_image = original_image / 255.0

        # Generate CAM
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=None)

        # grayscale_cam is [batch, height, width], take first image
        grayscale_cam = grayscale_cam[0, :]

        # Resize original image to match CAM if needed
        if original_image.shape[:2] != grayscale_cam.shape:
            import cv2
            original_image = cv2.resize(
                original_image,
                (grayscale_cam.shape[1], grayscale_cam.shape[0])
            )

        # Overlay CAM on image
        visualization = show_cam_on_image(
            original_image.astype(np.float32),
            grayscale_cam,
            use_rgb=True
        )

        return visualization


def create_simple_heatmap(
    probabilities: list,
    image_shape: tuple,
    tile_size: int = 224,
    stride: int = 112
) -> np.ndarray:
    """
    Create a simple probability heatmap from patch predictions.

    Args:
        probabilities: List of (prob, x, y, w, h) tuples
        image_shape: (height, width) of original image
        tile_size: Size of patches
        stride: Stride between patches

    Returns:
        Heatmap as numpy array [H, W]
    """
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # Fill heatmap with probabilities
    for prob, x, y, tw, th in probabilities:
        x_end = min(x + tw, w)
        y_end = min(y + th, h)
        heatmap[y:y_end, x:x_end] += prob
        counts[y:y_end, x:x_end] += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1)  # Avoid division by zero
    heatmap = heatmap / counts

    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = None
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image [H, W, 3] in [0, 255]
        heatmap: Heatmap [H, W] in [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        Overlaid visualization [H, W, 3]
    """
    import cv2

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    # Normalize heatmap to [0, 255]
    heatmap_normalized = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Resize heatmap to match image if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored,
            (image.shape[1], image.shape[0])
        )

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return result
